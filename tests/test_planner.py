"""规划器单元测试 — 不依赖 LLM"""
import json
import pytest
from agent.planner import Planner, SIMPLE_GOAL_KEYWORDS, MULTI_STEP_CUES
from agent.plan_schema import TaskType, TaskStatus, PlanStatus


class TestIsSimpleGoal:
    """is_simple_goal 复杂度判断"""

    def test_empty_goal(self):
        assert Planner().is_simple_goal("") is True

    def test_simple_list(self):
        assert Planner().is_simple_goal("列出当前目录文件") is True

    def test_simple_read(self):
        assert Planner().is_simple_goal("读取 config.yaml") is True

    def test_simple_view(self):
        assert Planner().is_simple_goal("查看文件内容") is True

    def test_multi_step_keyword(self):
        """包含多步关键词 → 需要规划"""
        assert Planner().is_simple_goal("先读取文件然后修改内容") is False
        assert Planner().is_simple_goal("列出目录并且分析结构") is False

    def test_long_text(self):
        """超过 30 字符 → 需要规划"""
        goal = "请帮我在项目中搜索所有使用了已废弃API的地方，然后逐一替换为新的API调用方式"
        assert len(goal) > 30
        assert Planner().is_simple_goal(goal) is False

    def test_no_keyword_short(self):
        """短文本但无简单关键词 → 不简单"""
        assert Planner().is_simple_goal("重构整个认证模块") is False


class TestInferTaskType:
    """_infer_task_type 任务类型推断"""

    def test_file_read(self):
        p = Planner()
        assert p._infer_task_type("读取文件内容") == TaskType.FILE_READ
        assert p._infer_task_type("列出目录") == TaskType.FILE_READ

    def test_file_write(self):
        p = Planner()
        assert p._infer_task_type("写入配置文件") == TaskType.FILE_WRITE
        assert p._infer_task_type("创建新文件") == TaskType.FILE_WRITE

    def test_analysis(self):
        p = Planner()
        assert p._infer_task_type("分析代码结构") == TaskType.ANALYSIS

    def test_verification(self):
        p = Planner()
        assert p._infer_task_type("验证执行结果") == TaskType.VERIFICATION
        assert p._infer_task_type("检查输出是否正确") == TaskType.VERIFICATION

    def test_default_command(self):
        p = Planner()
        assert p._infer_task_type("执行 npm install") == TaskType.COMMAND


class TestCreateMinimalPlan:
    """create_minimal_plan 单任务计划"""

    def test_basic(self):
        plan = Planner().create_minimal_plan("列出当前目录")
        assert plan.id == "plan_1"
        assert plan.goal == "列出当前目录"
        assert len(plan.tasks) == 1
        assert "task_1" in plan.tasks
        assert plan.tasks["task_1"].type == TaskType.FILE_READ
        assert plan.execution_order == ["task_1"]
        assert plan.status == PlanStatus.CREATED

    def test_counter_increments(self):
        p = Planner()
        p1 = p.create_minimal_plan("a")
        p2 = p.create_minimal_plan("b")
        assert p1.id == "plan_1"
        assert p2.id == "plan_2"


class TestParsePlan:
    """_parse_plan JSON 解析（多层容错）"""

    def _valid_json(self):
        return json.dumps({
            "summary": "测试计划",
            "tasks": [
                {"id": "1", "description": "读取文件", "type": "FILE_READ", "dependencies": []},
                {"id": "2", "description": "分析内容", "type": "ANALYSIS", "dependencies": ["1"]},
                {"id": "3", "description": "生成报告", "type": "FILE_WRITE", "dependencies": ["2"]},
            ],
        })

    def test_parse_valid_json(self):
        p = Planner()
        plan = p._parse_plan("p1", "测试目标", self._valid_json())
        assert plan.summary == "测试计划"
        assert len(plan.tasks) == 3
        # 重编号
        assert "task_1" in plan.tasks
        assert "task_2" in plan.tasks
        assert "task_3" in plan.tasks
        # 依赖重映射
        assert plan.tasks["task_2"].dependencies == ["task_1"]
        assert plan.tasks["task_3"].dependencies == ["task_2"]

    def test_parse_markdown_wrapped(self):
        """markdown 代码块包裹的 JSON"""
        raw = f"```json\n{self._valid_json()}\n```"
        p = Planner()
        plan = p._parse_plan("p1", "测试", raw)
        assert len(plan.tasks) == 3

    def test_parse_steps_alias(self):
        """兼容 steps 字段名"""
        raw = json.dumps({
            "summary": "步骤计划",
            "steps": [
                {"id": "s1", "description": "步骤一", "type": "ANALYSIS"},
            ],
        })
        p = Planner()
        plan = p._parse_plan("p1", "测试", raw)
        assert len(plan.tasks) == 1

    def test_parse_invalid_json_fallback(self):
        """无法解析 → 降级为单任务计划"""
        p = Planner()
        plan = p._parse_plan("p1", "测试目标", "这不是JSON")
        assert len(plan.tasks) == 1  # 降级为单任务

    def test_parse_empty_tasks_fallback(self):
        """tasks 为空数组 → 降级"""
        raw = json.dumps({"summary": "空", "tasks": []})
        p = Planner()
        plan = p._parse_plan("p1", "测试", raw)
        assert len(plan.tasks) == 1

    def test_parse_cyclic_dependency_fallback(self):
        """循环依赖 → 降级为单任务"""
        raw = json.dumps({
            "summary": "循环",
            "tasks": [
                {"id": "1", "description": "a", "dependencies": ["2"]},
                {"id": "2", "description": "b", "dependencies": ["1"]},
            ],
        })
        p = Planner()
        plan = p._parse_plan("p1", "测试", raw)
        # 循环依赖检测 → 降级
        assert len(plan.tasks) == 1

    def test_execution_order_set(self):
        """解析后 execution_order 通过拓扑排序生成"""
        p = Planner()
        plan = p._parse_plan("p1", "测试", self._valid_json())
        assert len(plan.execution_order) == 3
        # task_1 在 task_2 之前，task_2 在 task_3 之前
        assert plan.execution_order.index("task_1") < plan.execution_order.index("task_2")
        assert plan.execution_order.index("task_2") < plan.execution_order.index("task_3")

    def test_parse_task_type_unknown(self):
        """未知任务类型降级为 ANALYSIS"""
        raw = json.dumps({
            "summary": "",
            "tasks": [
                {"id": "1", "description": "未知类型", "type": "UNKNOWN_TYPE"},
            ],
        })
        p = Planner()
        plan = p._parse_plan("p1", "测试", raw)
        assert plan.tasks["task_1"].type == TaskType.ANALYSIS


class TestCleanJson:
    """_clean_json 清理 markdown"""

    def test_strip_code_block(self):
        p = Planner()
        assert p._clean_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strip_plain_code_block(self):
        p = Planner()
        assert p._clean_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_passthrough_clean(self):
        p = Planner()
        assert p._clean_json('{"a": 1}') == '{"a": 1}'
