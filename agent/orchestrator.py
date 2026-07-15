"""
Multi-Agent 编排子图 — Planner → Workers → Reviewer 协作

子图结构:
═══════════════════════════════════════════════════════════

  planner_agent_node (SubAgent PLANNER 生成 JSON 步骤)
    │  ├─ LLM 输出执行步骤 JSON
    │  └─ parse_step_json() 解析
    │
    ↓
  fan_out (条件边 — 循环入口)
    │
    ├─ 有可执行步骤? → Send("execute_step", {step_id}) × N 并行
    │                    │
    │                    ↓
    │               execute_step_node:
    │                 ├─ Worker SubAgent 执行 (ReAct + 工具)
    │                 │    └─ 工具调用走 _execute_tool → HITL 自动生效
    │                 ├─ Reviewer SubAgent 审查 (纯文本推理)
    │                 ├─ 通过 → 标记 COMPLETED
    │                 ├─ 不通过 → 注入反馈重试 (max 2)
    │                 └─ 重试耗尽 → interrupt("review_escalation") 人裁决
    │                    │
    │                    ↓  (所有 Send 实例收敛)
    │               fan_out (条件边 — 循环)
    │
    └─ 全部完成 → summarize_node → END
"""
import json
import logging
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt
from langgraph.graph.state import CompiledStateGraph
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import ensure_config

from .roles import AgentRole, AgentMessage, SubAgent
from .router import TEAM_PLANNER_PROMPT, TEAM_WORKER_PROMPT, TEAM_REVIEWER_PROMPT

logger = logging.getLogger(__name__)

from config import settings as _orch_config
# Reviewer 最大重试次数
MAX_RETRIES_PER_STEP = _orch_config.max_retries_per_step


class MultiAgentState(TypedDict):
    """Multi-Agent 子图状态

    关键字段:
    - steps: Annotated[dict, merge_steps] — 并行 Send 时自动合并步骤更新
    - current_step_id: Send 注入，标识当前执行哪个步骤
    """
    goal: str
    steps: Annotated[dict, merge_steps]    # {step_id: step_dict}
    current_step_id: str                   # Send 注入
    final_summary: str
    thread_id: str
    user_id: str


def merge_steps(left: dict, right: dict) -> dict:
    """LangGraph Annotated state reducer — 合并并行步骤状态更新

    策略: 非 PENDING 状态优先，防止旧状态覆盖已完成/失败的状态
    """
    merged = dict(left)
    for key, value in right.items():
        if key in merged:
            existing = merged[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                new_status = value.get("status", "PENDING")
                old_status = existing.get("status", "PENDING")
                if old_status == "PENDING" and new_status != "PENDING":
                    merged[key] = value
                elif new_status == "PENDING" and old_status != "PENDING":
                    pass  # 保留已完成/失败的状态
                else:
                    merged[key] = {**existing, **value}
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def build_multi_agent_graph(rag_service, checkpointer=None) -> CompiledStateGraph:
    """构建 Multi-Agent 子图

    三层角色协作:
    1. PLANNER: 分析用户需求 → 输出 JSON 步骤列表
    2. WORKER: 执行单个步骤 → ReAct 工具循环
    3. REVIEWER: 审查执行结果 → {approved, issues, suggestions}

    LangGraph 关键机制:
    - Send("execute_step", {"current_step_id": sid}): 为每个可执行步骤创建并行实例
    - merge_steps reducer: 并行实例返回的步骤状态自动合并
    - interrupt("review_escalation"): Reviewer 驳回超限后升级人裁决
    """

    async def planner_agent_node(state: MultiAgentState) -> dict:
        """节点 1: Planner SubAgent 生成执行步骤 JSON

        PLANNER 角色 → 纯文本推理（无工具） → 输出 JSON
        JSON 解析失败 → 直接返回错误汇总
        """
        planner = SubAgent(
            name="planner",
            role=AgentRole.PLANNER,
            llm=rag_service.light_llm,
            system_prompt=TEAM_PLANNER_PROMPT,
        )

        task_msg = AgentMessage.task(
            "orchestrator",
            f"请为以下任务制定执行计划：\n{state['goal']}",
        )
        logger.info("Multi-Agent Planner 开始规划: goal=%s...", state["goal"][:60])

        result = await planner.execute(task_msg)

        if result.type == "error":
            logger.error("Planner 执行失败: %s", result.content)
            return {"final_summary": f"[FAIL] 规划失败: {result.content}"}

        from .plan_schema import parse_plan_json
        steps = parse_plan_json(result.content)
        if not steps:
            logger.warning("无法解析 Planner 输出，原输出: %s", result.content[:300])
            return {
                "final_summary": (
                    "[FAIL] 无法解析执行计划，请尝试更具体的任务描述。\n"
                    f"原始输出:\n{result.content[:500]}"
                ),
            }

        logger.info("Multi-Agent 计划解析成功: %d 个步骤", len(steps))

        # 派发计划创建事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("plan_created", {
                "plan_id": "multi_agent",
                "summary": f"多 Agent 协作计划: {len(steps)} 个步骤",
                "task_count": len(steps),
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        return {"steps": {s["id"]: s for s in steps}}

    def fan_out(state: MultiAgentState):
        """条件边: 找出所有依赖已满足的 PENDING 步骤 → Send 扇出

        每次回到此节点时:
        1. 检查所有 PENDING 步骤的依赖是否全部 COMPLETED
        2. 将可执行步骤通过 Send 并行分发
        3. 无可执行步骤 → 进入汇总

        查找所有依赖已满足的 PENDING 步骤。
        """
        steps = state.get("steps", {})
        if not steps:
            logger.info("无步骤，跳转到 summarize")
            return "summarize"

        executable = _get_executable_step_ids(steps)

        if executable:
            logger.info("扇出 %d 个并行步骤: %s", len(executable), executable)
            # LangGraph Send 只把 payload 传给目标节点，不会带上主图 state，
            # 因此必须把 execute_step_node 需要的 steps / goal 一并放进 payload。
            goal = state.get("goal", "")
            return [Send("execute_step", {
                        "current_step_id": sid,
                        "steps": steps,
                        "goal": goal,
                    })
                    for sid in executable]

        # 无可执行步骤: 检查是否全部完成
        all_done = all(
            s.get("status") in ("COMPLETED", "FAILED", "SKIPPED")
            for s in steps.values()
        )
        if all_done:
            logger.info("全部步骤已处理，跳转到 summarize")
        else:
            pending = [sid for sid, s in steps.items()
                       if s.get("status") == "PENDING"]
            logger.warning("无可执行步骤但仍有 PENDING: %s (依赖未满足?)", pending)

        return "summarize"

    async def execute_step_node(state: MultiAgentState) -> dict:
        """节点 2: Worker 执行 + Reviewer 审查 + 重试循环

        1. Worker SubAgent 执行步骤 (带工具)
        2. Reviewer SubAgent 审查结果
        3. approved → 标记 COMPLETED
        4. rejected → 注入 issues 反馈 → 重试 (最多 2 次)
        5. 重试耗尽 → interrupt("review_escalation") → 人裁决

        工具调用经过 _execute_tool() → HITL 审批自动生效。
        """
        step_id = state["current_step_id"]
        step = dict(state["steps"].get(step_id, {}))
        if not step:
            logger.error("步骤 %s 不存在于 state 中", step_id)
            return {"steps": {step_id: {"status": "FAILED", "error": "步骤不存在"}}}

        logger.info("执行步骤: %s desc=%s", step_id, step.get("description", "")[:60])

        # 派发步骤开始事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("task_started", {
                "task_id": step_id,
                "description": step.get("description", ""),
                "type": step.get("type", "ANALYSIS"),
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        # 构建上下文（注入依赖步骤的已完成结果）
        context = _build_step_context(state.get("steps", {}), state["goal"], step)

        # Worker 实例 — 绑定工具（只有 WORKER 有工具）
        from .tools import search_knowledge, register_mcp_tools
        tools = [search_knowledge]
        if rag_service.mcp_registry.available_count > 0:
            tools.extend(register_mcp_tools(rag_service.mcp_registry))

        worker = SubAgent(
            name=f"worker-{step_id}",
            role=AgentRole.WORKER,
            llm=rag_service.react_llm,
            tools=tools,
            system_prompt=TEAM_WORKER_PROMPT,
        )

        # Reviewer 实例 — 无工具（纯文本推理）
        reviewer = SubAgent(
            name=f"reviewer-{step_id}",
            role=AgentRole.REVIEWER,
            llm=rag_service.light_llm,
            system_prompt=TEAM_REVIEWER_PROMPT,
        )

        # 执行 + 审查 + 重试循环
        enriched_content = context
        last_result = None
        last_issues = ""

        for retry in range(MAX_RETRIES_PER_STEP + 1):
            # Worker 执行
            task_msg = AgentMessage.task("orchestrator", enriched_content)
            result = await worker.execute_with_tools(
                task_msg, rag_service,
                thread_id=state.get("thread_id", "default"),
                user_id=state.get("user_id", "default"),
            )

            if result.type == "error":
                logger.warning("步骤 %s Worker 执行失败: %s", step_id, result.content)
                worker.clear_history()
                return {
                    "steps": {step_id: {
                        **step, "status": "FAILED", "error": result.content,
                    }},
                }

            # Reviewer 审查
            review_msg = AgentMessage.task(
                "orchestrator",
                f"原始任务：{step['description']}\n\n执行结果：\n{result.content}",
            )
            review_result = await reviewer.execute(review_msg)
            reviewer.clear_history()

            if review_result.type == "error":
                logger.warning("步骤 %s Reviewer 审查失败，保留当前结果", step_id)
                # 审查失败 → 保守策略：保留当前执行结果
                worker.clear_history()
                return {
                    "steps": {step_id: {
                        **step,
                        "status": "COMPLETED",
                        "result": result.content,
                    }},
                }

            approved = _parse_review_approval(review_result.content)

            if approved:
                logger.info("步骤 %s 审查通过 (retry=%d)", step_id, retry)
                worker.clear_history()

                # 派发步骤完成事件
                try:
                    _cfg = ensure_config()
                    await adispatch_custom_event("task_completed", {
                        "task_id": step_id,
                        "result_preview": (result.content or "")[:200],
                    }, config=_cfg)
                except Exception:
                    pass

                return {
                    "steps": {step_id: {
                        **step,
                        "status": "COMPLETED",
                        "result": result.content,
                    }},
                }

            # 不通过 → 注入反馈重试
            last_result = result
            last_issues = _parse_review_issues(review_result.content)
            logger.info("步骤 %s 审查不通过 (retry=%d): %s",
                         step_id, retry, last_issues[:100])

            enriched_content = (
                f"{context}\n\n"
                f"【审查反馈】之前的执行结果被审查拒绝，原因：\n{last_issues}\n"
                f"请修正问题后重新执行。"
            )

        # 重试耗尽 → 人裁决
        worker.clear_history()
        logger.info("步骤 %s 审查重试耗尽 (%d 次)，升级为人裁决", step_id, MAX_RETRIES_PER_STEP)

        # 派发审查升级事件
        try:
            _cfg = ensure_config()
            await adispatch_custom_event("review_escalation", {
                "step_id": step_id,
                "description": step["description"],
                "last_result": (last_result.content if last_result else "")[:300],
                "review_issues": last_issues[:300],
                "retries_exhausted": MAX_RETRIES_PER_STEP,
            }, config=_cfg)
        except RuntimeError:
            pass  # 非 stream 上下文
        except Exception as e:
            logger.debug("事件派发失败: %s", e)

        decision = interrupt({
            "type": "review_escalation",
            "hierarchy": "review",
            "step_id": step_id,
            "description": step["description"],
            "last_result": last_result.content if last_result else "",
            "review_issues": last_issues,
            "retries_exhausted": MAX_RETRIES_PER_STEP,
        })

        action = "skip"
        if isinstance(decision, dict):
            action = decision.get("action", "skip")

        logger.info("步骤 %s 人裁决: %s", step_id, action)

        if action == "accept_anyway":
            return {
                "steps": {step_id: {
                    **step,
                    "status": "COMPLETED",
                    "result": last_result.content if last_result else "人裁决: 接受",
                }},
            }
        elif action == "retry":
            # 重置为 PENDING，带着人的指引再试
            guidance = decision.get("guidance", "") if isinstance(decision, dict) else ""
            return {
                "steps": {step_id: {
                    **step,
                    "status": "PENDING",
                    "error": "",
                    "result": f"人裁决: 重试。指引: {guidance}" if guidance else "",
                }},
            }
        else:
            # skip
            return {
                "steps": {step_id: {**step, "status": "SKIPPED"}},
            }

    async def summarize_node(state: MultiAgentState) -> dict:
        """节点 3: 汇总所有步骤结果"""
        steps = state.get("steps", {})
        if not steps:
            return {"final_summary": state.get("final_summary", "无步骤结果")}

        ICONS = {
            "COMPLETED": "[OK]", "FAILED": "[FAIL]",
            "SKIPPED": "[SKIP]", "PENDING": "[···]", "RUNNING": "[>>]",
        }
        parts = ["多 Agent 协作执行总结："]

        for sid in sorted(steps.keys()):
            s = steps[sid]
            icon = ICONS.get(s.get("status", ""), "[??]")
            parts.append(f"{icon} [{sid}] {s.get('description', '')}")
            if s.get("result"):
                preview = s["result"][:300]
                if len(s["result"]) > 300:
                    preview += "..."
                parts.append(f"   结果：{preview}")
            if s.get("error"):
                parts.append(f"   错误：{s['error']}")

        all_ok = all(
            s.get("status") == "COMPLETED" for s in steps.values()
        )
        has_failures = any(
            s.get("status") == "FAILED" for s in steps.values()
        )

        if all_ok:
            status = "多 Agent 协作任务全部完成。"
        elif has_failures:
            status = "多 Agent 协作任务未完全完成，存在失败步骤。"
        else:
            status = "多 Agent 协作任务部分完成。"

        final = f"{status}\n\n" + "\n".join(parts)
        logger.info("Multi-Agent 汇总: %s (共 %d 个步骤)", status, len(steps))

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

        return {"final_summary": final}

    # ================================================================
    # 编译子图
    # ================================================================
    graph = StateGraph(MultiAgentState)

    graph.add_node("planner_agent", planner_agent_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("planner_agent")

    # planner → fan_out → execute_step (循环)
    graph.add_conditional_edges("planner_agent", fan_out, {
        "execute_step": "execute_step",
        "summarize": "summarize",
    })

    # execute_step → fan_out → (继续执行 / 汇总)
    graph.add_conditional_edges("execute_step", fan_out, {
        "execute_step": "execute_step",
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

def _get_executable_step_ids(steps: dict) -> list[str]:
    """返回所有依赖已满足的 PENDING 步骤 ID"""
    return [
        sid for sid, s in steps.items()
        if s.get("status") == "PENDING"
        and all(
            steps.get(dep, {}).get("status") == "COMPLETED"
            for dep in s.get("dependencies", [])
        )
    ]


def _build_step_context(steps: dict, goal: str, step: dict) -> str:
    """构建步骤上下文 — 注入依赖步骤的已完成结果"""
    parts = [f"总任务：{goal}"]

    # 注入依赖步骤的结果
    deps = step.get("dependencies", [])
    if deps:
        parts.append("\n依赖步骤的已完成结果：")
        for dep_id in deps:
            dep = steps.get(dep_id, {})
            status = dep.get("status", "UNKNOWN")
            if status == "COMPLETED":
                parts.append(f"\n[{dep_id}] {dep.get('description', '')} (已完成)")
                if dep.get("result"):
                    preview = dep["result"][:500]
                    if len(dep["result"]) > 500:
                        preview += "\n...(结果过长已截断)"
                    parts.append(f"结果：\n{preview}")
            else:
                parts.append(f"\n[{dep_id}] {dep.get('description', '')} (状态: {status})")
    else:
        parts.append("\n依赖步骤：无（可独立执行）")

    parts.append(f"\n当前任务：{step.get('description', '')}")
    parts.append(
        "\n请执行此步骤。如果任务描述中指定了工具名，请直接调用该工具。"
        "如果是 ANALYSIS 或 VERIFICATION 类型，请基于以上上下文直接给出结果。"
    )

    return "\n".join(parts)


def _parse_review_approval(content: str) -> bool:
    """解析 Reviewer 审批结果 — 默认不通过（保守策略）

    1. 尝试 JSON 解析 → approved 字段
    2. 失败 → 关键词匹配
    3. 默认 false（保守）
    """
    if not content:
        return False

    # 尝试 JSON 解析
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", content)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        data = json.loads(cleaned)
        approved = data.get("approved")
        if approved is not None:
            return bool(approved)
    except (json.JSONDecodeError, TypeError):
        pass

    # 关键词匹配
    lower = content.lower()
    has_negative = any(kw in lower for kw in [
        "未通过", "不通过", "不合格", "有问题",
        '"approved": false', '"approved":false',
    ])
    has_positive = any(kw in lower for kw in [
        "通过", "合格", '"approved": true', '"approved":true',
    ])

    if has_negative:
        return False
    if has_positive:
        return True

    # 无法判断 → 保守策略：不通过
    logger.warning("Reviewer 输出无法解析审批结果，默认不通过")
    return False


def _parse_review_issues(content: str) -> str:
    """解析 Reviewer 反馈的问题

    按优先级: issues > suggestions > summary
    """
    if not content:
        return "审查未通过，请改进执行结果"

    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", content)
        cleaned = re.sub(r"```\s*", "", cleaned).strip()
        data = json.loads(cleaned)

        # issues 数组
        issues = data.get("issues", [])
        if isinstance(issues, list) and issues:
            return "\n".join(f"- {i}" for i in issues if i)

        # suggestions 数组
        suggestions = data.get("suggestions", [])
        if isinstance(suggestions, list) and suggestions:
            return "\n".join(f"- {s}" for s in suggestions if s)

        # summary 字段
        summary = data.get("summary", "")
        if summary:
            return summary
    except (json.JSONDecodeError, TypeError):
        pass

    # 降级: 返回原始内容（截断）
    return content[:500] if len(content) > 500 else content
