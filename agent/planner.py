"""
规划器 — 使用轻量 LLM 将用户目标拆解为 ExecutionPlan

核心职责:
- is_simple_goal(): 判断是否跳过规划直接执行
- create_plan(): LLM 生成计划 + JSON 解析（多层容错）
- replan(): 失败后基于已完成任务重新规划
"""
import json
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from .plan_schema import ExecutionPlan, Task, TaskType, PlanStatus
from .router import PLANNER_PROMPT

logger = logging.getLogger(__name__)

# 简单目标关键词 — 跳过规划直接执行（单步操作类请求）
SIMPLE_GOAL_KEYWORDS = [
    "列出", "查看", "读取", "显示", "搜索", "当前目录",
    "文件", "ls", "cat", "dir",
    "审查", "分析", "创建", "删除", "写入", "修改",
    "帮我", "帮我查", "帮我找", "帮我看",
]
# 多步关键词 — 需要规划（需要多步分解的复杂请求）
MULTI_STEP_CUES = [
    "然后", "并且", "并", "再", "最后", "同时",
    "先", "之后", "接着", "以及", "还要",
]


class Planner:
    """LLM 任务规划器

    使用轻量 LLM (qwen-turbo) 将用户目标拆解为 ExecutionPlan。
    简单目标跳过 LLM 调用，直接创建单任务计划。
    """

    def __init__(self):
        self._plan_counter = 0

    def is_simple_goal(self, goal: str) -> bool:
        """判断是否跳过规划直接执行

        返回 True（简单）:
        - 含操作关键词（列出/读取/审查/帮我 等）→ ReAct 直接处理
        - 不含关键词但短文本（≤80字）→ ReAct 直接处理

        返回 False（复杂 → 升级 Plan）:
        - 不含操作关键词 且 超长文本（>80字）
        - 多步关键词（然后/并且）但没有操作关键词（无法判断意图）
        """
        if not goal:
            return True
        normalized = goal.strip()
        # 含操作关键词 → ReAct 直接处理（LLM 自己能处理多步）
        if any(kw in normalized for kw in SIMPLE_GOAL_KEYWORDS):
            return True
        # 不含关键词 且 超长 → Plan
        if len(normalized) > 80:
            return False
        # 多步关键词 → Plan（没有操作关键词参考，可能是复杂任务）
        if any(cue in normalized for cue in MULTI_STEP_CUES):
            return False
        # 短文本默认简单
        return True

    def create_minimal_plan(self, goal: str) -> ExecutionPlan:
        """简单目标跳过规划，创建单任务计划"""
        self._plan_counter += 1
        plan = ExecutionPlan(
            id=f"plan_{self._plan_counter}",
            goal=goal,
            summary=f"直接执行: {goal}",
        )
        task = Task(
            id="task_1",
            description=goal.strip(),
            type=self._infer_task_type(goal),
        )
        plan.tasks[task.id] = task
        plan.execution_order = [task.id]
        plan.status = PlanStatus.CREATED
        logger.info("简单目标跳过规划: %s", goal[:50])
        return plan

    def _infer_task_type(self, goal: str) -> TaskType:
        """推断简单任务类型——根据关键词匹配 FILE_READ/FILE_WRITE/ANALYSIS 等"""
        if any(kw in goal for kw in ["读取", "打开", "查看", "列出", "显示"]):
            return TaskType.FILE_READ
        if any(kw in goal for kw in ["写入", "修改", "创建", "生成"]):
            return TaskType.FILE_WRITE
        if any(kw in goal for kw in ["分析", "总结", "解释", "对比"]):
            return TaskType.ANALYSIS
        if any(kw in goal for kw in ["验证", "检查"]):
            return TaskType.VERIFICATION
        return TaskType.COMMAND

    async def create_plan(self, goal: str, llm) -> ExecutionPlan:
        """使用 LLM 生成执行计划

        流程:
        1. is_simple_goal() → 直接创建单任务计划
        2. 否则 LLM 生成 JSON → _parse_plan() 解析
        3. JSON 解析失败 → 降级为单任务计划
        """
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}"

        # 简单目标跳过 LLM 调用
        if self.is_simple_goal(goal):
            return self.create_minimal_plan(goal)

        logger.info("规划器调用 LLM 生成计划: goal=%s...", goal[:80])

        try:
            messages = [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=f"请为以下任务制定执行计划：\n{goal}"),
            ]
            response = await llm.ainvoke(messages)
            return self._parse_plan(plan_id, goal, response.content)
        except Exception as e:
            logger.error("LLM 规划失败: %s", e)
            return self.create_minimal_plan(goal)

    def _parse_plan(self, plan_id: str, goal: str, raw: str) -> ExecutionPlan:
        """解析 LLM 输出的计划 JSON — 多层容错

        委托 plan_schema.parse_plan_json() 做原始解析，本方法负责 Task 对象构建、
        循环依赖检测等 plan 层逻辑。
        """
        from .plan_schema import parse_plan_json

        plan = ExecutionPlan(id=plan_id, goal=goal)

        steps = parse_plan_json(raw, id_prefix="task_")
        if not steps:
            logger.warning("无法解析计划 JSON，降级为单任务。raw=%s", raw[:200])
            return self.create_minimal_plan(goal)

        # 解析 summary（从原始 JSON 中提取，parse_plan_json 不返回它）
        cleaned = self._clean_json(raw)
        summary = ""
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\"tasks\"[\s\S]*\}", cleaned)
            if not match:
                match = re.search(r"\{[\s\S]*\"steps\"[\s\S]*\}", cleaned)
            data = json.loads(match.group()) if match else {}
        summary = data.get("summary", "") if isinstance(data, dict) else ""
        plan.summary = summary

        # 构建 Task 对象
        for s in steps:
            task = Task(
                id=s["id"],
                description=s["description"],
                type=self._parse_task_type(s.get("type", "ANALYSIS")),
            )
            plan.tasks[s["id"]] = task

        # 建立依赖关系
        for s in steps:
            task = plan.tasks[s["id"]]
            for dep_id in s.get("dependencies", []):
                if dep_id in plan.tasks and dep_id != s["id"]:
                    task.dependencies.append(dep_id)
                    plan.tasks[dep_id].dependents.append(s["id"])

        # 校验循环依赖
        try:
            ts = TopologicalSorter({
                tid: set(t.dependencies) for tid, t in plan.tasks.items()
            })
            plan.execution_order = list(ts.static_order())
        except Exception as e:
            logger.warning("计划存在循环依赖: %s，降级为单任务", e)
            return self.create_minimal_plan(goal)

        plan.status = PlanStatus.CREATED
        logger.info(
            "计划解析成功: plan_id=%s tasks=%d batches=%d",
            plan_id, len(plan.tasks), len(plan.get_execution_batches()),
        )
        return plan

    def _parse_task_type(self, type_str: str) -> TaskType:
        """解析任务类型字符串，非法值降级为 ANALYSIS"""
        try:
            return TaskType(type_str.upper())
        except ValueError:
            return TaskType.ANALYSIS

    def _clean_json(self, raw: str) -> str:
        """清理 LLM 输出的 JSON: 去除 markdown 代码块、首尾空白"""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        return cleaned.strip()

    async def replan(
        self, failed_plan: ExecutionPlan,
        failure_reason: str, llm,
    ) -> ExecutionPlan:
        """失败后重新规划

        将已完成任务的结果注入上下文，让 LLM 避开已知问题。
        """
        context_parts = [
            f"原任务: {failed_plan.goal}",
            f"失败原因: {failure_reason}",
        ]

        completed = [
            t for t in failed_plan.tasks.values()
            if t.status == TaskStatus.COMPLETED
        ]
        if completed:
            context_parts.append("已完成的任务:")
            for t in completed:
                context_parts.append(f"- {t.id}: {t.description}")
                if t.result:
                    preview = t.result[:200]
                    context_parts.append(f"  结果: {preview}")

        context_parts.append("\n请制定新的执行计划，避开之前的问题。")
        new_goal = "\n".join(context_parts)

        logger.info("重规划: %s (已完成 %d/%d)",
                     failure_reason[:50], len(completed), len(failed_plan.tasks))
        return await self.create_plan(new_goal, llm)


# 导入拓扑排序器（放在文件末尾避免 forward reference）
from graphlib import TopologicalSorter
