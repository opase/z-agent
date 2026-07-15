"""
执行计划数据结构 — Task DAG + 拓扑排序

核心设计:
- Task: 单个可执行任务单元，支持依赖/被依赖关系
- ExecutionPlan: 任务 DAG，拓扑排序确定执行顺序
- merge_tasks: LangGraph Annotated state reducer，合并并行 Send 的状态更新
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter


class TaskType(str, Enum):
    """任务类型"""
    PLANNING = "PLANNING"        # 规划任务
    FILE_READ = "FILE_READ"      # 读取文件
    FILE_WRITE = "FILE_WRITE"    # 写入文件
    COMMAND = "COMMAND"           # 执行命令
    ANALYSIS = "ANALYSIS"        # 分析结果
    VERIFICATION = "VERIFICATION" # 验证结果


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "PENDING"      # 等待执行
    RUNNING = "RUNNING"      # 执行中
    COMPLETED = "COMPLETED"  # 已完成
    FAILED = "FAILED"        # 失败
    SKIPPED = "SKIPPED"      # 跳过


class PlanStatus(str, Enum):
    """计划状态"""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    """单个可执行任务单元

    字段说明:
    - id, description, type, status, result, error
    - dependencies: 前置任务 ID 列表
    - dependents: 依赖此任务的后置任务 ID 列表
    """
    id: str
    description: str
    type: TaskType = TaskType.ANALYSIS
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        """序列化为字典（存储到 LangGraph state 中）"""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type.value,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        """从字典反序列化"""
        return cls(
            id=d.get("id", ""),
            description=d.get("description", ""),
            type=TaskType(d.get("type", "ANALYSIS")),
            dependencies=list(d.get("dependencies", [])),
            dependents=list(d.get("dependents", [])),
            status=TaskStatus(d.get("status", "PENDING")),
            result=d.get("result", ""),
            error=d.get("error", ""),
        )


@dataclass
class ExecutionPlan:
    """执行计划 DAG

    核心能力:
    - 维护 Task 字典 + 拓扑排序执行顺序
    - get_executable_tasks(): 查找依赖全部满足的 PENDING 任务
    - get_execution_batches(): 拓扑分批（同批可并行）
    - get_progress(): 完成百分比
    """
    id: str
    goal: str
    summary: str = ""
    tasks: dict[str, Task] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.CREATED

    def get_executable_tasks(self) -> list[Task]:
        """返回所有依赖已满足的 PENDING 任务

        一个任务可执行的条件:
        1. 自身状态为 PENDING
        2. 所有 dependencies 中的任务状态都是 COMPLETED
        """
        return [
            t for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
            and all(
                dep_id in self.tasks
                and self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in t.dependencies
            )
        ]

    def get_execution_batches(self) -> list[list[Task]]:
        """按拓扑排序分批，同批内所有任务可并行执行

        使用 Python 标准库 graphlib.TopologicalSorter:
        - ts.get_ready() 返回当前所有依赖已满足的节点
        - 每批处理完后调用 ts.done() 释放下一批

        Returns:
            [[batch1_task1, batch1_task2], [batch2_task1], ...]
        """
        if not self.tasks:
            return []
        ts = TopologicalSorter({
            tid: set(t.dependencies) for tid, t in self.tasks.items()
        })
        ts.prepare()
        batches = []
        while ts.is_active():
            batch_ids = list(ts.get_ready())
            batch = [self.tasks[tid] for tid in batch_ids if tid in self.tasks]
            if batch:
                batches.append(batch)
            for tid in batch_ids:
                ts.done(tid)
        return batches

    def get_progress(self) -> float:
        """已完成任务占比 0.0 ~ 1.0"""
        if not self.tasks:
            return 1.0
        completed = sum(
            1 for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED
        )
        return completed / len(self.tasks)

    def is_all_completed(self) -> bool:
        """全部任务是否已完成"""
        return all(
            t.status == TaskStatus.COMPLETED
            for t in self.tasks.values()
        )

    def has_failed(self) -> bool:
        """是否有任务失败"""
        return any(
            t.status == TaskStatus.FAILED
            for t in self.tasks.values()
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "id": self.id,
            "goal": self.goal,
            "summary": self.summary,
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "execution_order": list(self.execution_order),
            "status": self.status.value,
            "progress": self.get_progress(),
        }


import json as _json
import logging as _logging
import re as _re

_logger = _logging.getLogger(__name__)


def parse_plan_json(raw: str, id_prefix: str = "step_") -> list[dict]:
    """解析 planner LLM 输出的步骤/任务 JSON —— 多层容错，供 planner 和 orchestrator 共用。

    期望格式:
    {"summary": "...", "steps": [{ "id": "step_1", "description": "...",
      "type": "ANALYSIS", "dependencies": [] }]}
    也兼容 "tasks" 字段。

    容错策略:
    1. 清理 markdown 代码块
    2. 尝试直接 JSON.parse
    3. 失败 → 正则提取 JSON 片段
    4. 仍失败 → 返回空列表

    步骤重编号: 原始 id → {id_prefix}1, {id_prefix}2, ...
    依赖重映射: 原始依赖 id → 新编号
    """
    if not raw:
        return []

    # 1. 清理 markdown
    cleaned = _re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = _re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    # 2. 解析 JSON
    data = None
    try:
        data = _json.loads(cleaned)
    except _json.JSONDecodeError:
        # 3. 正则提取
        match = _re.search(r"\{[\s\S]*\"steps\"[\s\S]*\}", cleaned)
        if not match:
            match = _re.search(r"\{[\s\S]*\"tasks\"[\s\S]*\}", cleaned)
        if match:
            try:
                data = _json.loads(match.group())
            except _json.JSONDecodeError:
                pass

    if data is None:
        _logger.warning("无法解析步骤 JSON: %s", raw[:200])
        return []

    raw_steps = data.get("steps", data.get("tasks", []))
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        _logger.warning("JSON 无 steps/tasks 数组")
        return []

    # 重编号 + 依赖重映射
    id_mapping: dict[str, str] = {}
    steps = []
    for i, s in enumerate(raw_steps, 1):
        original_id = str(s.get("id", f"{id_prefix}{i}"))
        new_id = f"{id_prefix}{i}"
        id_mapping[original_id] = new_id
        steps.append({
            "id": new_id,
            "description": str(s.get("description", "")),
            "type": str(s.get("type", "ANALYSIS")),
            "dependencies": [],
            "status": "PENDING",
            "result": "",
            "error": "",
        })

    for i, s in enumerate(raw_steps, 1):
        deps = s.get("dependencies", [])
        if isinstance(deps, list):
            mapped_deps = []
            for dep_id in deps:
                mapped = id_mapping.get(str(dep_id), str(dep_id))
                if mapped != f"{id_prefix}{i}":
                    mapped_deps.append(mapped)
            steps[i - 1]["dependencies"] = mapped_deps

    _logger.info("解析 %d 个步骤 (重编号)", len(steps))
    return steps


def merge_tasks(left: dict, right: dict) -> dict:
    """LangGraph Annotated state reducer — 合并并行任务的状态更新

    当多个 Send("execute_task") 并发执行并返回 {"tasks": {task_id: ...}} 时，
    LangGraph 自动调用此函数合并状态。

    合并策略:
    - 两个 dict 中不重叠的 key 直接合并
    - 重叠的 key 取较新的状态（非 PENDING 优先）
    """
    merged = dict(left)
    for key, value in right.items():
        if key in merged:
            existing = merged[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                # 状态优先级: 不要用旧状态覆盖新状态
                # 两边的 status 不同时，取非 PENDING 的那个
                new_status = value.get("status", "PENDING")
                old_status = existing.get("status", "PENDING")
                if old_status == "PENDING" and new_status != "PENDING":
                    merged[key] = value  # 新状态覆盖旧状态
                elif new_status == "PENDING" and old_status != "PENDING":
                    pass  # 保留旧状态（已完成/失败的不被覆盖）
                else:
                    merged[key] = {**existing, **value}
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged
