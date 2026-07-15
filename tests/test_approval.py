"""审批流程测试"""
import pytest
from datetime import datetime, timedelta
from core.approval_store import ApprovalStore
from core.session_store import SQLiteStore


class TestApprovalStore:
    """审批记录持久化 CRUD 测试"""

    def test_create_and_get(self, tmp_path):
        """创建并查询审批记录"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        approval_id = "test-id-001"
        store.create(
            approval_id=approval_id,
            user_id="user_a",
            thread_id="thread_1",
            tool_name="mcp__demo__echo",
            tool_args={"text": "hello"},
            server_name="demo",
            expires_at=(datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        )

        record = store.get(approval_id)
        assert record is not None
        assert record["user_id"] == "user_a"
        assert record["thread_id"] == "thread_1"
        assert record["status"] == "pending"
        assert record["tool_name"] == "mcp__demo__echo"
        assert record["tool_args"] == {"text": "hello"}

    def test_resolve_approved(self, tmp_path):
        """审批通过"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        store.create(
            "id-001", "user_a", "thread_1", "mcp__demo__add",
            {"a": 1, "b": 2}, "demo",
            (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        success = store.resolve("id-001", "approved")
        assert success
        record = store.get("id-001")
        assert record["status"] == "approved"

    def test_resolve_rejected(self, tmp_path):
        """审批拒绝"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        store.create(
            "id-001", "user_a", "thread_1", "mcp__demo__rm",
            {"path": "/tmp/x"}, "demo",
            (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        success = store.resolve("id-001", "rejected", "op1", "测试拒绝")
        assert success
        record = store.get("id-001")
        assert record["status"] == "rejected"
        assert record["reject_reason"] == "测试拒绝"

    def test_idempotent_resolve(self, tmp_path):
        """幂等性：已处理的审批不能再次修改"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        store.create(
            "id-001", "user_a", "thread_1", "mcp__demo__echo",
            {"text": "x"}, "demo",
            (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        # 第一次成功
        assert store.resolve("id-001", "approved")
        # 第二次失败（幂等）
        assert not store.resolve("id-001", "rejected")

    def test_expire_pending(self, tmp_path):
        """过期审批自动标记"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        # 创建一个已过期的审批
        store.create(
            "id-expired", "user_a", "thread_1", "mcp__demo__echo",
            {"text": "x"}, "demo",
            (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
        )
        count = store.expire_pending()
        assert count == 1
        record = store.get("id-expired")
        assert record["status"] == "expired"

    def test_pending_for_thread(self, tmp_path):
        """按线程查询待处理审批"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        store.create("id-1", "user_a", "thread_a", "t1", {}, "srv", "2099-01-01 00:00:00")
        store.create("id-2", "user_a", "thread_a", "t2", {}, "srv", "2099-01-01 00:00:00")
        store.create("id-3", "user_a", "thread_b", "t3", {}, "srv", "2099-01-01 00:00:00")

        pending_a = store.get_pending_for_thread("thread_a")
        assert len(pending_a) == 2

        first = store.get_pending_first("thread_a")
        assert first["id"] == "id-1"

    def test_mark_interrupted(self, tmp_path):
        """服务重启时标记中断"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)

        store.create("id-1", "user_a", "thread_x", "t1", {}, "srv", "2099-01-01 00:00:00")
        store.create("id-2", "user_a", "thread_x", "t2", {}, "srv", "2099-01-01 00:00:00")

        count = store.mark_interrupted_for_thread("thread_x")
        assert count == 2
        assert store.get("id-1")["status"] == "interrupted"
        assert store.get("id-2")["status"] == "interrupted"

    def test_not_found(self, tmp_path):
        """查询不存在的审批"""
        db_path = str(tmp_path / "test_approval.db")
        store = ApprovalStore(db_path)
        assert store.get("nonexistent") is None
        assert store.get_pending_first("no_thread") is None


class TestApprovalTableExists:
    """验证审批表在 SQLiteStore 中自动创建"""

    def test_table_auto_created(self, tmp_path):
        """SQLiteStore 初始化时 approval_requests 表应该存在"""
        import os
        db_path = str(tmp_path / "test.db")
        store = SQLiteStore(db_path)

        # 直接查询表是否存在
        conn = store._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='approval_requests'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None, "approval_requests 表未创建"
