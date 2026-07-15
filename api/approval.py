"""审批 API 路由 — HITL 审批恢复 + SSE 结构化事件

工具审批: asyncio.Event 原地等待，resolve() 内 signal event 即恢复。
计划审批/审查升级: LangGraph interrupt() + Command(resume=...) 兜底。
"""
import json
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langgraph.types import Command

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/approval", tags=["审批"])


class ApprovalResult(BaseModel):
    """审批决定请求体

    支持四种 decision 值:
    - approved: 批准当前操作（工具调用 / 计划 / 步骤 / 审查升级）
    - approve_all: 批准当前 + 后续同线程 MCP 工具自动批准
    - rejected: 拒绝当前操作（可附带原因供 LLM 重试）
    """
    user_id: str = "default"
    decision: str  # "approved" | "rejected" | "approve_all"
    reject_reason: str | None = None


def _get_rag(request: Request):
    return request.app.state.rag


@router.post("/{thread_id}/resume")
async def resume_approval(thread_id: str, body: ApprovalResult, request: Request):
    """审批恢复端点

    工具审批: resolve() → signal_decision() → Event 唤醒工具继续执行。
    计划/审查审批: Command(resume=...) 恢复 LangGraph interrupt。
    """
    rag = _get_rag(request)

    # 校验
    if body.decision not in ("approved", "rejected", "approve_all"):
        raise HTTPException(400, "decision 必须为 approved / approve_all / rejected")

    decision_val = "approved" if body.decision in ("approved", "approve_all") else "rejected"

    # 批准所有: 设置线程级自动批准标记
    if body.decision == "approve_all":
        rag.approval_mgr.set_approve_all(thread_id)

    # 持久化审批记录 + signal event（工具审批通过 Event 机制自动继续）
    pending = rag.approval_mgr.get_pending(thread_id)
    if pending:
        rag.approval_mgr.resolve(
            approval_id=pending["id"],
            user_id=body.user_id,
            decision=body.decision,
            reject_reason=body.reject_reason,
            operator_id=body.user_id,
        )

    # 兜底: LangGraph Command(resume=...) 仅用于 plan_review / review_escalation
    # 工具审批走 Event 方案时 graph 不处于 interrupt 状态，跳过此步。
    answer = ""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state_snapshot = await rag.agent_graph.aget_state(config)
        if state_snapshot and state_snapshot.next and state_snapshot.interrupts:
            # 获取中断负载，判断类型以构造正确的 resume 数据
            item = state_snapshot.interrupts[0]
            interrupt_payload = getattr(item, "value", item)
            interrupt_type = ""
            if isinstance(interrupt_payload, dict):
                interrupt_type = interrupt_payload.get("type", "")

            # 根据中断类型构造 resume 数据
            if interrupt_type == "review_escalation":
                # Multi-Agent 审查升级：approved → accept_anyway, rejected → skip
                resume_data = {
                    "decision": decision_val,
                    "action": "accept_anyway" if decision_val == "approved" else "skip",
                }
                if body.reject_reason:
                    resume_data["reject_reason"] = body.reject_reason
            elif interrupt_type == "plan_review":
                # 计划审批：approved → 继续, rejected → cancel
                resume_data = {
                    "decision": decision_val,
                    "action": "cancel" if decision_val == "rejected" else "approved",
                }
                if body.reject_reason:
                    resume_data["feedback"] = body.reject_reason
            else:
                # 兜底：工具审批等（理论上走不到这里，Event 机制已处理）
                resume_data = {"decision": decision_val}
                if body.reject_reason:
                    resume_data["reject_reason"] = body.reject_reason

            final_result = await rag.agent_graph.ainvoke(
                Command(resume=resume_data), config,
            )
            output = final_result.get("final_output", final_result)
            if isinstance(output, dict):
                answer = output.get("answer", "")
            else:
                answer = str(output)
    except Exception:
        pass  # aget_state 可能因并发等抛异常，忽略

    logger.info("审批恢复成功: thread=%s decision=%s answer_chars=%d",
                thread_id, decision_val, len(answer))

    return {
        "status": decision_val,
        "message": (
            "审批已通过"
            if decision_val == "approved"
            else "审批已拒绝"
        ),
        "answer": answer,
    }
