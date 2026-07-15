"""
线程状态查询 API — 前端刷新页面后恢复审批 UI

GET /state/{thread_id}
  查询当前线程的等待状态，返回:
  - status: "awaiting_approval" | "idle"
  - interrupts: 等待中的审批列表（类型、层级、ID）
  - next_nodes: 下一个将执行的图节点

前端使用场景:
1. 页面刷新后恢复审批弹窗
2. 长时间等待后查询是否超时
3. SSE 断线重连后获取当前状态
"""
import logging
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/state", tags=["状态"])


@router.get("/{thread_id}")
async def get_thread_state(thread_id: str, request: Request):
    """查询当前线程的等待状态

    前端刷新页面后，调用此接口判断是否需要显示审批弹窗。
    返回的 interrupts 列表中包含审批类型、层级、ID 等前端所需信息。
    """
    rag = request.app.state.rag
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = rag.agent_graph.get_state(config)
    except Exception as e:
        logger.warning("查询状态失败: thread=%s error=%s", thread_id, e)
        return {
            "thread_id": thread_id,
            "status": "error",
            "message": str(e),
        }

    interrupts = []
    if hasattr(state, "interrupts") and state.interrupts:
        for item in state.interrupts:
            # LangGraph Interrupt 对象: 取 .value 属性获取 payload
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                interrupts.append(value)
            else:
                interrupts.append({"type": "unknown", "raw": str(value)})

    next_nodes = []
    if hasattr(state, "next") and state.next:
        next_nodes = list(state.next)

    # 确定状态
    if interrupts:
        status = "awaiting_approval"
    elif next_nodes:
        status = "running"
    else:
        status = "idle"

    return {
        "thread_id": thread_id,
        "status": status,
        "interrupts": interrupts,
        "next_nodes": next_nodes,
    }


@router.get("/{thread_id}/approval")
async def get_pending_approval(thread_id: str, request: Request):
    """查询线程的待处理审批详情

    返回数据库中持久化的审批请求信息。
    """
    rag = request.app.state.rag

    try:
        pending = rag.approval_mgr.get_pending(thread_id)
    except Exception as e:
        logger.warning("查询审批失败: thread=%s error=%s", thread_id, e)
        return {"thread_id": thread_id, "pending": None, "error": str(e)}

    if not pending:
        return {"thread_id": thread_id, "pending": None}

    return {
        "thread_id": thread_id,
        "pending": {
            "approval_id": pending.get("id"),
            "tool": pending.get("tool_name"),
            "args": pending.get("tool_args"),
            "server": pending.get("server_name"),
            "status": pending.get("status"),
            "created_at": pending.get("created_at"),
            "expires_at": pending.get("expires_at"),
        },
    }
