"""
Plan-and-Execute 子图 — 使用 LangGraph Send 动态扇出实现任务并行

子图结构:
═══════════════════════════════════════════════════════════

  planner_node (LLM 生成 ExecutionPlan)
    │  ├─ 可选: interrupt("plan_review") 人审计划
    │
    ↓
  fan_out (条件边 — 循环入口)
    │
    ├─ 有可执行任务? → Send("execute_task", {task_id}) × N 并行
    │                    │
    │                    ↓
    │               execute_task_node (单任务 ReAct 循环)
    │                    │  ├─ _execute_tool() → interrupt tool 审批
    │                    │  └─ 完成 → 更新 task 状态
    │                    │
    │                    ↓  (所有 Send 实例收敛)
    │               fan_out (条件边 — 循环)
    │
    └─ 全部完成或无法继续 → summarize_node → END
"""
import logging
import json
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt
from langgraph.graph.state import CompiledStateGraph
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import ensure_config

from .plan_schema import (
    ExecutionPlan, Task, TaskType, TaskStatus, PlanStatus, merge_tasks,
)
from .planner import Planner
from .router import PLAN_TASK_PROMPT

logger = logging.getLogger(__name__)

from config import settings as _plan_config
# 每个 Task 的 ReAct 最大迭代次数
MAX_TASK_ITERATIONS = _plan_config.max_task_iterations
# 失败后触发重规划的进度阈值
REPLAN_THRESHOLD = _plan_config.replan_threshold


class PlanExecuteState(TypedDict):
    """Plan-and-Execute 子图状态

    关键字段:
    - goal: 用户目标（必填，不要用 state.get("goal", "") 而用 state.get）
    - tasks: Annotated[dict, merge_tasks] — 并行 Send 时自动合并任务更新
    - current_task_id: Send 注入，标识当前执行的是哪个任务
    - plan_review_enabled: 是否在计划生成后暂停等人审
    - thread_id/user_id: 传递给 _execute_tool 用于 HITL 审批
    """
    goal: str                                      # 用户目标
    plan_dict: dict                                # ExecutionPlan.to_dict()
    current_task_id: str                           # 当前正在执行的任务 ID
    final_summary: str                             # 最终汇总文本
    plan_approved: bool                            # 计划是否已通过人审
    plan_review_enabled: bool                      # 是否启用计划审批
    tasks: Annotated[dict, merge_tasks]            # {task_id: task_dict}
    thread_id: str                                 # 用于 HITL 审批
    user_id: str                                   # 用于 HITL 审批


def build_plan_execute_graph(rag_service, checkpointer=None) -> CompiledStateGraph:
    """构建 Plan-and-Execute 子图

    此子图作为主图的一个节点使用:
    - 输入: goal (str)
    - 输出: final_summary (str)

    LangGraph 关键机制:
    - Send("execute_task", {"current_task_id": tid}): 为每个可执行任务创建并行实例
    - merge_tasks reducer: 并行实例返回的 tasks 字典自动合并
    - interrupt(): 工具审批在 _execute_tool 中触发，计划审批在 planner_node 中触发
    """

    async def planner_node(state: PlanExecuteState) -> dict:
        """节点 1: 使用 LLM 生成执行计划

        如果 plan_review_enabled=True，生成后暂停等人审计划结构。
        人可以选择: approved / supplement (带补充反馈重新规划) / cancel
        """
        planner = Planner()
        plan = await planner.create_plan(state.get("goal", ""), rag_service.light_llm)
        logger.info("Plan-and-Execute 计划生成: %d 个任务, %d 个批次",
                     len(plan.tasks), len(plan.get_execution_batches()))

        # 可选: 计划级 HITL
        if state.get("plan_review_enabled", False):
            decision = interrupt({
                "type": "plan_review",
                "hierarchy": "plan",
                "plan": plan.to_dict(),
            })
            if isinstance(decision, dict):
                action = decision.get("action", "cancel")
                if action == "cancel":
                    logger.info("计划被用户取消")
                    return {
                        "plan_dict": None,
                        "plan_approved": False,
                        "final_summary": "[CANCELLED] 计划已取消",
                    }
                if action == "supplement":
                    feedback = decision.get("feedback", "")
                    logger.info("计划需要补充: %s", feedback[:80])
                    plan = await planner.create_plan(
                        f"{state.get('goal', '')}\n补充要求：{feedback}",
                        rag_service.light_llm,
                    )

        plan.status = PlanStatus.RUNNING

        # 派发计划创建事件（astream_events 可捕获）
        try:
            _cfg = ensure_config()
            # 将 Task 对象转为可序列化的 task id 列表
            batches = plan.get_execution_batches()
            serializable_batches = [[t.id for t in batch] for batch in batches]
            await adispatch_custom_event("plan_created", {
                "plan_id": plan.id,
                "summary": plan.summary,
                "task_count": len(plan.tasks),
                "execution_order": serializable_batches,
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)  # 非 astream_events 上下文时忽略

        return {
            "plan_dict": plan.to_dict(),
            "plan_approved": True,
            "tasks": {t.id: t.to_dict() for t in plan.tasks.values()},
            "final_summary": "",
        }

    def fan_out(state: PlanExecuteState):
        """条件边: 查找可执行任务 → Send 扇出

        每次执行后回到此节点判断:
        - 有依赖已满足的 PENDING 任务 → Send 并行执行
        - 全部完成 / 取消 / 无法继续 → summarize

        这是实现 DAG 拓扑执行的核心: 利用 LangGraph 的 Send 机制，
        每次只执行当前"就绪"的任务，执行完后自然收敛回到此节点。
        """
        plan_dict = state.get("plan_dict")
        if not plan_dict or not state.get("plan_approved", True):
            logger.info("计划无效或未批准，跳转到 summarize")
            return "summarize"

        plan = _dict_to_plan(plan_dict, state.get("tasks", {}))
        executable = plan.get_executable_tasks()

        if executable:
            logger.info("扇出 %d 个并行任务: %s",
                         len(executable),
                         [t.id for t in executable])
            # LangGraph Send 只把 payload 传给目标节点，不会带上主图 state，
            # 因此必须把 execute_task_node 需要的上下文一并放进 payload。
            return [Send("execute_task", {
                        "current_task_id": t.id,
                        "tasks": state.get("tasks", {}),
                        "plan_dict": plan_dict,
                        "goal": state.get("goal", ""),
                        "thread_id": state.get("thread_id", "default"),
                        "user_id": state.get("user_id", "default"),
                    })
                    for t in executable]

        # 无可执行任务: 检查原因
        if plan.is_all_completed():
            logger.info("全部任务完成，跳转到 summarize")
        elif plan.has_failed():
            logger.info("有任务失败，跳转到 summarize")
        else:
            # 有 PENDING 但依赖未满足 → 不应该出现（依赖完成就会可执行）
            pending = [tid for tid, t in plan.tasks.items()
                       if t.status == TaskStatus.PENDING]
            logger.warning("无可执行任务但仍有 PENDING: %s (可能循环依赖)", pending)

        return "summarize"

    async def execute_task_node(state: PlanExecuteState) -> dict:
        """节点 2: 执行单个 Task 的 ReAct 循环

        每个 Task 独立执行:
        1. 构建上下文（注入依赖任务的结果）
        2. 运行 ReAct 循环: LLM 自主决定工具调用或输出结果
        3. 工具调用经过 _execute_tool() → HITL 审批自动生效
        4. 返回 task 状态更新（merge_tasks reducer 自动合并）

        如果失败且整体进度 < 50%，触发 replan。
        """
        task_id = state["current_task_id"]
        task_dict = state.get("tasks", {}).get(task_id, {})
        task = Task.from_dict(task_dict)
        plan_dict = state.get("plan_dict", {})
        plan = _dict_to_plan(plan_dict, state.get("tasks", {}))

        logger.info("执行任务: %s type=%s desc=%s",
                     task_id, task.type.value, task.description[:60])

        # 派发任务开始事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("task_started", {
                "task_id": task_id,
                "description": task.description,
                "type": task.type.value,
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        # 构建任务上下文
        context = _build_task_context(state.get("tasks", {}),
                                      state.get("goal", ""), task)

        # System prompt — PLAN 模式，注入任务类型和描述
        system_text = PLAN_TASK_PROMPT.format(
            task_type=task.type.value,
            task_description=task.description,
        )
        # skill 索引注入 system prompt
        from skill.index_formatter import format_skill_index
        skill_index = format_skill_index(rag_service.skill_registry.enabled_skills())
        if skill_index:
            system_text += f"\n\n{skill_index}"
        system = SystemMessage(content=system_text)

        # 准备工具
        from .tools import search_knowledge, register_mcp_tools
        from skill.tool import load_skill as load_skill_tool, list_skills as list_skills_tool
        tools = [search_knowledge]
        has_mcp = rag_service.mcp_registry.available_count > 0
        if has_mcp:
            mcp_tools = register_mcp_tools(rag_service.mcp_registry)
            tools.extend(mcp_tools)
        tools.extend([load_skill_tool, list_skills_tool])

        llm_with_tools = rag_service.react_llm.bind_tools(tools)
        messages = [system, HumanMessage(content=context)]

        # ReAct 循环
        from memory.token_budget import compact_react_messages
        for iteration in range(MAX_TASK_ITERATIONS):
            # 调用 LLM 前按需压缩上下文
            messages = await compact_react_messages(messages, rag_service.light_llm)
            response = await llm_with_tools.ainvoke(messages)

            if response.tool_calls:
                # 工具调用
                messages.append(response)
                for tc in response.tool_calls:
                    from .graph import _execute_tool
                    result = await _execute_tool(
                        tc, rag_service,
                        thread_id=state.get("thread_id", "default"),
                        user_id=state.get("user_id", "default"),
                    )
                    messages.append(ToolMessage(
                        content=result, tool_call_id=tc["id"],
                    ))
                logger.info("Task %s 第 %d 轮: %d 个工具调用",
                             task_id, iteration + 1, len(response.tool_calls))
            else:
                # 任务完成 — LLM 输出最终结果
                task.status = TaskStatus.COMPLETED
                task.result = response.content or ""
                logger.info("Task %s 完成: result_chars=%d",
                             task_id, len(task.result))

                # 派发任务完成事件
                try:
                    _cfg = ensure_config()
                    await adispatch_custom_event("task_completed", {
                        "task_id": task_id,
                        "result_preview": task.result[:200],
                    }, config=_cfg)
                except Exception:
                    pass

                return {"tasks": {task_id: task.to_dict()}}

        # 超过最大迭代: 检查是否需要 replan
        task.status = TaskStatus.FAILED
        task.error = f"超过最大迭代次数 ({MAX_TASK_ITERATIONS})"
        logger.warning("Task %s 失败: %s", task_id, task.error)

        # 派发任务失败事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("task_failed", {
                "task_id": task_id,
                "error": task.error,
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        # 进度 < 50% 时触发重规划
        if plan.get_progress() < REPLAN_THRESHOLD:
            logger.info("进度 %.0f%% < %.0f%%，触发重规划",
                         plan.get_progress() * 100, REPLAN_THRESHOLD * 100)
            planner = Planner()
            try:
                new_plan = await planner.replan(plan, task.error,
                                                rag_service.light_llm)
                new_tasks = {t.id: t.to_dict() for t in new_plan.tasks.values()}
                logger.info("重规划完成: %d 个新任务", len(new_tasks))
                return {
                    "plan_dict": new_plan.to_dict(),
                    "tasks": new_tasks,
                    "final_summary": "",
                }
            except Exception as e:
                logger.error("重规划失败: %s", e)

        return {"tasks": {task_id: task.to_dict()}}

    async def summarize_node(state: PlanExecuteState) -> dict:
        """节点 3: 汇总所有任务结果"""
        tasks = {
            tid: Task.from_dict(td)
            for tid, td in state.get("tasks", {}).items()
        }
        if not tasks:
            return {"final_summary": state.get("final_summary", "无任务结果")}

        plan_dict = state.get("plan_dict", {})
        plan = _dict_to_plan(plan_dict, state.get("tasks", {}))

        # 任务状态图标
        ICONS = {
            TaskStatus.COMPLETED: "[OK]",
            TaskStatus.FAILED: "[FAIL]",
            TaskStatus.SKIPPED: "[SKIP]",
            TaskStatus.PENDING: "[···]",
            TaskStatus.RUNNING: "[>>]",
        }

        parts = [f"计划执行完成: {plan_dict.get('summary', '')}"]

        for tid in sorted(tasks.keys()):
            t = tasks[tid]
            icon = ICONS.get(t.status, "[??]")
            parts.append(f"{icon} [{tid}] {t.description}")
            if t.result:
                preview = t.result[:300]
                if len(t.result) > 300:
                    preview += "..."
                parts.append(f"   {preview}")
            elif t.error:
                parts.append(f"   错误: {t.error}")

        all_completed = all(
            t.status == TaskStatus.COMPLETED for t in tasks.values()
        )
        has_any_failed = any(
            t.status == TaskStatus.FAILED for t in tasks.values()
        )

        if all_completed:
            status_line = "计划全部完成。"
        elif has_any_failed:
            status_line = "计划部分完成，有任务失败。"
        else:
            status_line = "计划未完全执行。"

        final = f"{status_line}\n\n" + "\n".join(parts)
        logger.info("Plan-and-Execute 汇总: %s (共 %d 个任务)",
                     status_line, len(tasks))

        # 派发计划完成事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("plan_completed", {
                "summary": final[:500],
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        return {"final_summary": final, "plan_dict": plan.to_dict()}

    # ================================================================
    # 编译子图
    # ================================================================
    graph = StateGraph(PlanExecuteState)

    graph.add_node("planner", planner_node)
    graph.add_node("execute_task", execute_task_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("planner")

    # planner → fan_out → execute_task (循环)
    graph.add_conditional_edges("planner", fan_out, {
        "execute_task": "execute_task",
        "summarize": "summarize",
    })

    # execute_task → fan_out → (继续执行 / 汇总)
    graph.add_conditional_edges("execute_task", fan_out, {
        "execute_task": "execute_task",
        "summarize": "summarize",
    })

    graph.add_edge("summarize", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ================================================================
# 辅助函数
# ================================================================

def _build_task_context(
    tasks_dict: dict, goal: str, task: Task,
) -> str:
    """构建任务上下文 — 注入依赖任务结果

    包含总目标、依赖任务的描述和结果、当前任务描述。
    """
    parts = [f"总目标：{goal}"]
    parts.append(f"当前任务：{task.description}")

    if task.dependencies:
        parts.append("\n依赖任务的执行结果：")
        for dep_id in task.dependencies:
            dep_dict = tasks_dict.get(dep_id, {})
            dep = Task.from_dict(dep_dict)
            status_label = {
                TaskStatus.COMPLETED: "已完成",
                TaskStatus.FAILED: "失败",
                TaskStatus.SKIPPED: "已跳过",
            }.get(dep.status, dep.status.value)
            parts.append(f"\n[{dep_id}] {dep.description} ({status_label})")
            if dep.result:
                # 限制过长结果注入上下文
                result_preview = dep.result[:500]
                if len(dep.result) > 500:
                    result_preview += "\n...(结果过长已截断)"
                parts.append(f"结果：\n{result_preview}")
    else:
        parts.append("\n依赖任务：无（可独立执行）")

    parts.append(
        "\n请执行此任务。如果任务描述中指定了工具名，请直接调用该工具。"
        "如果是 ANALYSIS 或 VERIFICATION 类型，请基于以上上下文直接给出结果。"
    )
    return "\n".join(parts)


def _dict_to_plan(plan_dict: dict, tasks_dict: dict) -> ExecutionPlan:
    """从字典恢复 ExecutionPlan 对象"""
    plan = ExecutionPlan(
        id=plan_dict.get("id", ""),
        goal=plan_dict.get("goal", ""),
        summary=plan_dict.get("summary", ""),
        status=PlanStatus(plan_dict.get("status", "RUNNING")),
        execution_order=list(plan_dict.get("execution_order", [])),
    )
    plan.tasks = {
        tid: Task.from_dict(td) for tid, td in tasks_dict.items()
    }
    return plan
