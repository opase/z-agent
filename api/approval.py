"""审批 API 路由 — HITL 审批恢复 + SSE 结构化事件

工具审批 / 计划审批 / 审查升级统一通过 LangGraph interrupt() + Command(resume=...) 恢复。
"""
import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langgraph.types import Command

from config import settings as config
from core import tracing

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
    """审批恢复端点。

    当前线程必须处于 LangGraph interrupt 状态。接口先更新审批审计记录，
    再通过 Command(resume=...) 从 checkpoint 恢复图执行。
    """
    rag = _get_rag(request)

    # 校验
    if body.decision not in ("approved", "rejected", "approve_all"):
        raise HTTPException(400, "decision 必须为 approved / approve_all / rejected")

    decision_val = "approved" if body.decision in ("approved", "approve_all") else "rejected"
    graph_config = {"configurable": {"thread_id": thread_id}}

    state_snapshot = await rag.agent_graph.aget_state(graph_config)
    if not state_snapshot or not state_snapshot.interrupts:
        logger.warning(
            "审批恢复失败: thread=%s 无可恢复中断 snapshot=%s next=%s interrupts=%s",
            thread_id,
            bool(state_snapshot),
            getattr(state_snapshot, "next", None) if state_snapshot else None,
            getattr(state_snapshot, "interrupts", None) if state_snapshot else None,
        )
        raise HTTPException(409, "当前线程没有待恢复的审批中断")

    item = state_snapshot.interrupts[0]
    interrupt_payload = getattr(item, "value", item)
    interrupt_type = ""
    if isinstance(interrupt_payload, dict):
        interrupt_type = interrupt_payload.get("type", "")

    if interrupt_type == "approval_required":
        approval_id = interrupt_payload.get("approval_id") if isinstance(interrupt_payload, dict) else None
        if not approval_id:
            raise HTTPException(500, "工具审批中断缺少 approval_id")
        resolved = rag.approval_mgr.resolve(
            approval_id=approval_id,
            user_id=body.user_id,
            decision=body.decision,
            reject_reason=body.reject_reason,
            operator_id=body.user_id,
        )
        if not resolved.get("success"):
            raise HTTPException(409, resolved.get("message", "审批状态更新失败"))
        resume_data = {"decision": decision_val}
        if body.reject_reason:
            resume_data["reject_reason"] = body.reject_reason
    elif interrupt_type == "review_escalation":
        resume_data = {
            "decision": decision_val,
            "action": "accept_anyway" if decision_val == "approved" else "skip",
        }
        if body.reject_reason:
            resume_data["reject_reason"] = body.reject_reason
    elif interrupt_type == "plan_review":
        resume_data = {
            "decision": decision_val,
            "action": "cancel" if decision_val == "rejected" else "approved",
        }
        if body.reject_reason:
            resume_data["feedback"] = body.reject_reason
    else:
        raise HTTPException(409, f"不支持的审批中断类型: {interrupt_type or 'unknown'}")

    # 取原始问题/模式（interrupt 前的 checkpoint state），让 resume 阶段共享首次请求的
    # 业务语义（user.query / chat.mode / 消融标签）。否则 HITL 的 resume 会变成一条
    # 没有归属的孤儿 trace —— 回答阶段的 token / 延迟在 Phoenix 里查不到归属。
    orig_state = getattr(state_snapshot, "values", None)
    question = (orig_state.get("question", "") if isinstance(orig_state, dict) else "") or ""
    mode = (orig_state.get("mode", "react") if isinstance(orig_state, dict) else "react") or "react"

    with tracing.span(
        "z_agent.chat",
        **{
            "user.query": tracing.sanitize_query(question),
            "chat.mode": mode,
            "chat.has_image": False,
            "chat.stream": False,
            "chat.resume": True,  # 区分首次请求 vs 审批恢复，Phoenix 可 group-by
            "chat.thread_id": thread_id,
        },
    ) as rspan:
        tracing.set_io(rspan, kind=tracing.KIND_CHAIN, input_value=question)
        tracing.set_ablation(rspan)

        final_result = await rag.agent_graph.ainvoke(
            Command(resume=resume_data), graph_config,
        )

        # 二次 interrupt（本次审批后又触发新的审批）：收尾 span 后返回
        if isinstance(final_result, dict) and "__interrupt__" in final_result:
            interrupt_info = final_result["__interrupt__"]
            item = interrupt_info[0] if isinstance(interrupt_info, list) and interrupt_info else interrupt_info
            next_payload = getattr(item, "value", item)
            nxt_type = next_payload.get("type", "") if isinstance(next_payload, dict) else ""
            tracing.set_io(rspan, output_value=f"[已暂停等待审批: {nxt_type}]")
            return {
                "status": "interrupted",
                "message": "审批已处理，等待下一次审批",
                "answer": "",
                "interrupt": next_payload,
            }

        output = final_result.get("final_output", final_result)
        answer = output.get("answer", "") if isinstance(output, dict) else str(output)
        tracing.set_io(rspan, output_value=answer or (output.get("status", "") if isinstance(output, dict) else ""))
        if config.eval_semantic_enabled and answer.strip():
            ctx = output.get("context", "") if isinstance(output, dict) else ""
            await rag._record_semantic_eval(rspan, question, answer, ctx)

    sessions = getattr(request.app.state, "sessions", None)
    turn_count = 1
    if sessions and answer:
        session = sessions.get(thread_id, body.user_id)
        if session:
            question = final_result.get("question", "") if isinstance(final_result, dict) else ""
            if question:
                session.memory.add_message("user", question)
            session.memory.add_message("assistant", answer)
            turn_count = session.memory.turn_count

    logger.info("审批恢复成功: thread=%s type=%s decision=%s answer_chars=%d",
                thread_id, interrupt_type, decision_val, len(answer))

    return {
        "status": decision_val,
        "message": "审批已通过" if decision_val == "approved" else "审批已拒绝",
        "answer": answer,
        "turn_count": turn_count,
    }
