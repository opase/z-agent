"""Plan-and-Execute 数据结构与合并逻辑单元测试"""
import pytest
from agent.plan_schema import (
    Task, TaskType, TaskStatus, ExecutionPlan, PlanStatus, merge_tasks,
)


class TestTask:
    """Task 序列化/反序列化"""

    def test_to_dict_round_trip(self):
        """to_dict → from_dict 往返一致"""
        t = Task(
            id="t1", description="读取配置", type=TaskType.FILE_READ,
            dependencies=["t0"], status=TaskStatus.COMPLETED,
            result="配置内容...", error="",
        )
        d = t.to_dict()
        t2 = Task.from_dict(d)
        assert t2.id == "t1"
        assert t2.type == TaskType.FILE_READ
        assert t2.status == TaskStatus.COMPLETED
        assert t2.dependencies == ["t0"]
        assert t2.result == "配置内容..."

    def test_from_dict_defaults(self):
        """空字典反序列化使用默认值"""
        t = Task.from_dict({})
        assert t.id == ""
        assert t.type == TaskType.ANALYSIS
        assert t.status == TaskStatus.PENDING
        assert t.dependencies == []


class TestExecutionPlan:
    """ExecutionPlan DAG 操作"""

    def _make_plan(self):
        """构造 3 任务 DAG: t1 → t3, t2 → t3"""
        plan = ExecutionPlan(id="p1", goal="测试目标", summary="测试")
        plan.tasks = {
            "t1": Task(id="t1", description="任务1", type=TaskType.ANALYSIS),
            "t2": Task(id="t2", description="任务2", type=TaskType.FILE_READ),
            "t3": Task(id="t3", description="任务3", type=TaskType.VERIFICATION,
                       dependencies=["t1", "t2"]),
        }
        return plan

    def test_get_executable_tasks_initial(self):
        """初始状态: t1 和 t2 可执行（无依赖），t3 不可（依赖未满足）"""
        plan = self._make_plan()
        exe = plan.get_executable_tasks()
        ids = {t.id for t in exe}
        assert ids == {"t1", "t2"}

    def test_get_executable_tasks_after_partial_complete(self):
        """t1 完成后: 只有 t2 可执行（t3 仍被 t2 阻塞）"""
        plan = self._make_plan()
        plan.tasks["t1"].status = TaskStatus.COMPLETED
        exe = plan.get_executable_tasks()
        ids = {t.id for t in exe}
        assert ids == {"t2"}

    def test_get_executable_tasks_all_deps_met(self):
        """t1 和 t2 都完成后: t3 可执行"""
        plan = self._make_plan()
        plan.tasks["t1"].status = TaskStatus.COMPLETED
        plan.tasks["t2"].status = TaskStatus.COMPLETED
        exe = plan.get_executable_tasks()
        assert len(exe) == 1
        assert exe[0].id == "t3"

    def test_get_executable_tasks_empty_when_all_done(self):
        """全部完成后无可执行任务"""
        plan = self._make_plan()
        for t in plan.tasks.values():
            t.status = TaskStatus.COMPLETED
        assert plan.get_executable_tasks() == []

    def test_get_execution_batches(self):
        """拓扑分批: 第1批 [t1, t2], 第2批 [t3]"""
        plan = self._make_plan()
        batches = plan.get_execution_batches()
        assert len(batches) == 2
        batch1_ids = {t.id for t in batches[0]}
        batch2_ids = {t.id for t in batches[1]}
        assert batch1_ids == {"t1", "t2"}
        assert batch2_ids == {"t3"}

    def test_get_execution_batches_empty(self):
        """空计划返回空批次"""
        plan = ExecutionPlan(id="empty", goal="")
        assert plan.get_execution_batches() == []

    def test_get_progress(self):
        """进度计算"""
        plan = self._make_plan()
        assert plan.get_progress() == 0.0
        plan.tasks["t1"].status = TaskStatus.COMPLETED
        assert abs(plan.get_progress() - 1/3) < 0.01
        plan.tasks["t2"].status = TaskStatus.COMPLETED
        assert abs(plan.get_progress() - 2/3) < 0.01
        plan.tasks["t3"].status = TaskStatus.COMPLETED
        assert plan.get_progress() == 1.0

    def test_is_all_completed(self):
        plan = self._make_plan()
        assert not plan.is_all_completed()
        for t in plan.tasks.values():
            t.status = TaskStatus.COMPLETED
        assert plan.is_all_completed()

    def test_has_failed(self):
        plan = self._make_plan()
        assert not plan.has_failed()
        plan.tasks["t1"].status = TaskStatus.FAILED
        assert plan.has_failed()

    def test_to_dict(self):
        plan = self._make_plan()
        d = plan.to_dict()
        assert d["id"] == "p1"
        assert d["goal"] == "测试目标"
        assert len(d["tasks"]) == 3
        assert d["status"] == "CREATED"
        assert d["progress"] == 0.0


class TestMergeTasks:
    """merge_tasks reducer — 并行 Send 状态合并"""

    def test_disjoint_keys(self):
        """不重叠的 key 直接合并"""
        left = {"t1": {"status": "COMPLETED", "result": "ok"}}
        right = {"t2": {"status": "PENDING"}}
        merged = merge_tasks(left, right)
        assert "t1" in merged
        assert "t2" in merged

    def test_new_status_overrides_pending(self):
        """新状态（非 PENDING）覆盖旧的 PENDING"""
        left = {"t1": {"status": "PENDING"}}
        right = {"t1": {"status": "COMPLETED", "result": "done"}}
        merged = merge_tasks(left, right)
        assert merged["t1"]["status"] == "COMPLETED"
        assert merged["t1"]["result"] == "done"

    def test_pending_does_not_override_completed(self):
        """PENDING 不覆盖已完成的 COMPLETED"""
        left = {"t1": {"status": "COMPLETED", "result": "ok"}}
        right = {"t1": {"status": "PENDING"}}
        merged = merge_tasks(left, right)
        assert merged["t1"]["status"] == "COMPLETED"
        assert merged["t1"]["result"] == "ok"

    def test_both_non_pending_merge(self):
        """双方都非 PENDING 时浅合并"""
        left = {"t1": {"status": "COMPLETED", "result": "old"}}
        right = {"t1": {"status": "COMPLETED", "result": "new", "extra": "x"}}
        merged = merge_tasks(left, right)
        assert merged["t1"]["result"] == "new"
        assert merged["t1"]["extra"] == "x"

    def test_non_dict_values(self):
        """非 dict 值直接覆盖"""
        left = {"key": "old"}
        right = {"key": "new"}
        merged = merge_tasks(left, right)
        assert merged["key"] == "new"
