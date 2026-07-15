"""审批管理器 — 管理 MCP 工具调用的 HITL 审批请求生命周期"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta

from config import settings as config

logger = logging.getLogger(__name__)


class ApprovalManager:
    """审批请求的创建、查询、解析和过期管理

    使用 asyncio.Event 实现原地等待审批决议：
    - 工具调用前创建审批请求 + Event
    - await event.wait() 原地等待（不重启节点函数）
    - 前端审批后 → signal event → 工具继续执行
    - 消息历史自然保留，无需缓存/恢复逻辑
    """

    def __init__(self):
        self._store = None  # ApprovalStore 实例
        self._timeout_minutes = config.approval_timeout_minutes
        # 批准所有标记: {thread_id: bool}
        self._approve_all_threads: set[str] = set()
        # asyncio.Event 机制: {approval_id: asyncio.Event}
        self._events: dict[str, asyncio.Event] = {}
        # 审批决议结果: {approval_id: dict}
        self._decisions: dict[str, dict] = {}
        # 审批创建时间: {approval_id: float}
        self._created_times: dict[str, float] = {}

    def set_store(self, store):
        self._store = store

    def is_approve_all(self, thread_id: str) -> bool:
        return thread_id in self._approve_all_threads

    def set_approve_all(self, thread_id: str) -> None:
        self._approve_all_threads.add(thread_id)

    def create_request(
        self, user_id: str, thread_id: str, tool_name: str,
        tool_args: dict, server_name: str, approval_id: str | None = None,
    ) -> str:
        approval_id = approval_id or str(uuid.uuid4())
        expires_at = (datetime.now() + timedelta(minutes=self._timeout_minutes)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        if self._store:
            self._store.create(
                approval_id=approval_id, user_id=user_id, thread_id=thread_id,
                tool_name=tool_name, tool_args=tool_args, server_name=server_name,
                expires_at=expires_at,
            )
        # 创建 Event 供 await_decision 等待
        self._events[approval_id] = asyncio.Event()
        self._created_times[approval_id] = time.time()
        # 指标: 审批创建
        from core import metrics
        metrics.approval_created.labels(tool_name=tool_name, server=server_name).inc()
        logger.info("审批请求已创建: id=%s tool=%s thread=%s user=%s",
                     approval_id, tool_name, thread_id, user_id)
        return approval_id

    async def await_decision(self, approval_id: str, timeout: float = 60.0) -> dict:
        """原地等待审批决议

        返回决议字典: {"decision": "approved"|"rejected", "reject_reason": str|None}
        超时后返回 {"decision": "expired"}
        """
        event = self._events.get(approval_id)
        if not event:
            return {"decision": "expired", "reject_reason": "审批请求不存在"}

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("审批超时: %s", approval_id)
            self._decisions[approval_id] = {"decision": "expired", "reject_reason": "审批超时"}
            self._events.pop(approval_id, None)
            return {"decision": "expired", "reject_reason": "审批超时"}

        decision = self._decisions.pop(approval_id, {"decision": "expired"})
        self._events.pop(approval_id, None)
        # 指标: 审批等待时长 + 决议
        created = self._created_times.pop(approval_id, None)
        from core import metrics
        if created:
            metrics.approval_wait_duration.observe(time.time() - created)
        metrics.approval_decisions.labels(decision=decision.get("decision", "expired")).inc()
        return decision

    def signal_decision(self, approval_id: str, decision: str,
                        reject_reason: str | None = None) -> bool:
        """通知等待中的工具：审批决议已到达"""
        event = self._events.get(approval_id)
        if not event:
            return False
        self._decisions[approval_id] = {
            "decision": decision,
            "reject_reason": reject_reason,
        }
        event.set()
        return True

    def resolve(self, approval_id: str, user_id: str, decision: str,
                reject_reason: str | None = None, operator_id: str | None = None) -> dict:
        """同步决议（Graph 路径 / 纯 API 调用）"""
        if decision not in ("approved", "rejected", "approve_all"):
            return {"success": False, "message": "decision 无效", "current_status": "unknown"}
        if not self._store:
            return {"success": False, "message": "存储未初始化", "current_status": "unknown"}

        record = self._store.get(approval_id)
        if not record:
            return {"success": False, "message": "审批请求不存在", "current_status": "not_found"}
        if record["user_id"] != user_id:
            return {"success": False, "message": "无权操作此审批请求", "current_status": record["status"]}
        if record["status"] != "pending":
            return {"success": False, "message": "该审批已处理", "current_status": record["status"]}

        # 检查过期
        expires_at = record.get("expires_at", "")
        if expires_at and datetime.now().strftime("%Y-%m-%d %H:%M:%S") >= expires_at:
            self._store.resolve(approval_id, "expired", operator_id, "审批超时")
            return {"success": False, "message": "审批已超时", "current_status": "expired"}

        # 批准所有：记录 thread_id
        if decision == "approve_all":
            self._approve_all_threads.add(record["thread_id"])
            decision = "approved"

        updated = self._store.resolve(approval_id, decision, operator_id, reject_reason)
        if updated:
            logger.info("审批 %s → %s (by %s)", approval_id, decision, operator_id or user_id)
            # 通知等待中的工具继续执行
            self.signal_decision(approval_id, decision, reject_reason)
            return {"success": True, "message": f"审批已{'通过' if decision == 'approved' else '拒绝'}", "current_status": decision}
        return {"success": False, "message": "审批状态更新失败", "current_status": "unknown"}

    def get_pending(self, thread_id: str) -> dict | None:
        if self._store:
            return self._store.get_pending_first(thread_id)
        return None

    def expire_check(self, approval_id: str) -> bool:
        if not self._store:
            return False
        record = self._store.get(approval_id)
        if not record or record["status"] != "pending":
            return False
        expires_at = record.get("expires_at", "")
        return expires_at and datetime.now().strftime("%Y-%m-%d %H:%M:%S") >= expires_at
