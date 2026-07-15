"""Multi-Agent 编排辅助函数单元测试"""
import json
import pytest
from agent.orchestrator import (
    _get_executable_step_ids,
    _parse_review_approval, _parse_review_issues, merge_steps,
)
from agent.plan_schema import parse_plan_json


class TestParseStepJson:
    """parse_plan_json — Planner JSON 解析"""

    def test_valid_json(self):
        raw = json.dumps({
            "summary": "测试计划",
            "steps": [
                {"id": "1", "description": "读取", "type": "FILE_READ", "dependencies": []},
                {"id": "2", "description": "分析", "type": "ANALYSIS", "dependencies": ["1"]},
            ],
        })
        steps = parse_plan_json(raw)
        assert len(steps) == 2
        assert steps[0]["id"] == "step_1"
        assert steps[1]["id"] == "step_2"
        assert steps[1]["dependencies"] == ["step_1"]

    def test_markdown_wrapped(self):
        raw = '```json\n' + json.dumps({
            "steps": [{"id": "s1", "description": "任务", "type": "ANALYSIS"}],
        }) + '\n```'
        steps = parse_plan_json(raw)
        assert len(steps) == 1

    def test_tasks_alias(self):
        """兼容 tasks 字段名"""
        raw = json.dumps({
            "tasks": [{"id": "t1", "description": "任务", "type": "COMMAND"}],
        })
        steps = parse_plan_json(raw)
        assert len(steps) == 1

    def test_invalid_json(self):
        steps = parse_plan_json("这不是有效的JSON")
        assert steps == []

    def test_empty_steps_array(self):
        raw = json.dumps({"steps": []})
        steps = parse_plan_json(raw)
        assert steps == []

    def test_renumbering(self):
        """步骤重编号: 原始 id → step_1, step_2..."""
        raw = json.dumps({
            "steps": [
                {"id": "alpha", "description": "a", "type": "ANALYSIS"},
                {"id": "beta", "description": "b", "type": "ANALYSIS"},
                {"id": "gamma", "description": "c", "type": "ANALYSIS"},
            ],
        })
        steps = parse_plan_json(raw)
        assert [s["id"] for s in steps] == ["step_1", "step_2", "step_3"]

    def test_dependency_remapping(self):
        """依赖 ID 重映射到新编号"""
        raw = json.dumps({
            "steps": [
                {"id": "x", "description": "a", "type": "ANALYSIS"},
                {"id": "y", "description": "b", "type": "ANALYSIS", "dependencies": ["x"]},
            ],
        })
        steps = parse_plan_json(raw)
        assert steps[1]["dependencies"] == ["step_1"]

    def test_self_dependency_removed(self):
        """自依赖被移除"""
        raw = json.dumps({
            "steps": [
                {"id": "a", "description": "a", "type": "ANALYSIS", "dependencies": ["a"]},
            ],
        })
        steps = parse_plan_json(raw)
        assert steps[0]["dependencies"] == []

    def test_default_status_pending(self):
        raw = json.dumps({
            "steps": [{"id": "1", "description": "t", "type": "ANALYSIS"}],
        })
        steps = parse_plan_json(raw)
        assert steps[0]["status"] == "PENDING"


class TestGetExecutableStepIds:
    """_get_executable_step_ids — 可执行步骤查找"""

    def test_all_pending_no_deps(self):
        steps = {
            "step_1": {"status": "PENDING", "dependencies": []},
            "step_2": {"status": "PENDING", "dependencies": []},
        }
        result = _get_executable_step_ids(steps)
        assert set(result) == {"step_1", "step_2"}

    def test_blocked_by_dependency(self):
        steps = {
            "step_1": {"status": "PENDING", "dependencies": []},
            "step_2": {"status": "PENDING", "dependencies": ["step_1"]},
        }
        result = _get_executable_step_ids(steps)
        assert result == ["step_1"]

    def test_dep_completed_unblocks(self):
        steps = {
            "step_1": {"status": "COMPLETED", "dependencies": []},
            "step_2": {"status": "PENDING", "dependencies": ["step_1"]},
        }
        result = _get_executable_step_ids(steps)
        assert result == ["step_2"]

    def test_all_completed_empty(self):
        steps = {
            "step_1": {"status": "COMPLETED", "dependencies": []},
            "step_2": {"status": "COMPLETED", "dependencies": ["step_1"]},
        }
        result = _get_executable_step_ids(steps)
        assert result == []

    def test_empty_steps(self):
        assert _get_executable_step_ids({}) == []


class TestParseReviewApproval:
    """_parse_review_approval — Reviewer 审批解析"""

    def test_json_approved_true(self):
        raw = json.dumps({"approved": True, "summary": "ok"})
        assert _parse_review_approval(raw) is True

    def test_json_approved_false(self):
        raw = json.dumps({"approved": False, "issues": ["问题1"]})
        assert _parse_review_approval(raw) is False

    def test_keyword_positive(self):
        assert _parse_review_approval("审查结果：通过，执行质量良好") is True

    def test_keyword_negative(self):
        assert _parse_review_approval("审查结果：未通过，存在问题") is False

    def test_empty_string(self):
        assert _parse_review_approval("") is False

    def test_none(self):
        assert _parse_review_approval(None) is False

    def test_ambiguous_defaults_false(self):
        """无法判断 → 保守不通过"""
        assert _parse_review_approval("这是一段普通文本") is False

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"approved": true}\n```'
        assert _parse_review_approval(raw) is True


class TestParseReviewIssues:
    """_parse_review_issues — Reviewer 反馈解析"""

    def test_json_issues_array(self):
        raw = json.dumps({"issues": ["问题1", "问题2"], "approved": False})
        result = _parse_review_issues(raw)
        assert "问题1" in result
        assert "问题2" in result

    def test_json_suggestions_fallback(self):
        raw = json.dumps({"suggestions": ["建议1"], "approved": False})
        result = _parse_review_issues(raw)
        assert "建议1" in result

    def test_json_summary_fallback(self):
        raw = json.dumps({"summary": "总体不合格", "approved": False})
        result = _parse_review_issues(raw)
        assert "总体不合格" in result

    def test_plain_text_fallback(self):
        result = _parse_review_issues("执行结果不够完整")
        assert "执行结果不够完整" in result

    def test_empty_string(self):
        result = _parse_review_issues("")
        assert "审查未通过" in result


class TestMergeSteps:
    """merge_steps reducer — 与 merge_tasks 策略一致"""

    def test_disjoint_keys(self):
        left = {"s1": {"status": "COMPLETED"}}
        right = {"s2": {"status": "PENDING"}}
        merged = merge_steps(left, right)
        assert "s1" in merged and "s2" in merged

    def test_new_overrides_pending(self):
        left = {"s1": {"status": "PENDING"}}
        right = {"s1": {"status": "COMPLETED", "result": "done"}}
        merged = merge_steps(left, right)
        assert merged["s1"]["status"] == "COMPLETED"

    def test_pending_no_override_completed(self):
        left = {"s1": {"status": "COMPLETED", "result": "ok"}}
        right = {"s1": {"status": "PENDING"}}
        merged = merge_steps(left, right)
        assert merged["s1"]["status"] == "COMPLETED"
