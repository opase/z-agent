"""MCP 客户端单元测试"""
import json
import pytest
from mcp_client.schema import normalize_tool_schema
from mcp_client.registry import MCPToolRegistry


class TestSchemaNormalize:
    """Schema 裁剪/规范化测试"""

    def test_passthrough_simple_schema(self):
        """简单 Schema 透传无变更"""
        raw = {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "输入文本"}},
            "required": ["text"],
        }
        result = normalize_tool_schema(raw)
        assert result["type"] == "object"
        assert "text" in result["properties"]
        assert result["required"] == ["text"]

    def test_strip_unsupported_keywords(self):
        """移除不支持的 JSON Schema 关键字"""
        raw = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": 0, "maximum": 100, "default": 50}
            },
        }
        result = normalize_tool_schema(raw)
        prop = result["properties"]["x"]
        assert "minimum" not in prop
        assert "maximum" not in prop
        assert "default" not in prop

    def test_force_top_level_object(self):
        """确保顶级 type 为 object"""
        raw = {"type": "string", "description": "not an object"}
        result = normalize_tool_schema(raw)
        assert result["type"] == "object"

    def test_empty_schema_gets_properties(self):
        """空 Schema 自动补充 properties"""
        raw = {}
        result = normalize_tool_schema(raw)
        assert result["type"] == "object"
        assert "properties" in result

    def test_oneof_flatten_first(self):
        """oneOf 取第一个选项"""
        raw = {
            "oneOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "number"}}},
            ]
        }
        result = normalize_tool_schema(raw)
        assert "a" in result.get("properties", {})

    def test_defs_ref_resolution(self):
        """$defs/$ref 引用展开"""
        raw = {
            "type": "object",
            "properties": {"addr": {"$ref": "#/$defs/address"}},
            "$defs": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        }
        result = normalize_tool_schema(raw)
        addr = result["properties"].get("addr", {})
        assert addr.get("properties", {}).get("city", {}).get("type") == "string"


class TestRegistryNamespace:
    """工具注册表命名空间测试"""

    def test_register_format(self):
        """命名空间格式 mcp__{server}__{tool}"""
        registry = MCPToolRegistry()
        full_name = registry.register(
            "demo", "echo", "回显文本", {"type": "object", "properties": {}},
            server_approval_mode="require_approval",
        )
        assert full_name == "mcp__demo__echo"

    def test_no_name_collision(self):
        """不同服务器的同名工具不冲突"""
        registry = MCPToolRegistry()
        n1 = registry.register("srv_a", "do", "A的do", {"type": "object", "properties": {}})
        n2 = registry.register("srv_b", "do", "B的do", {"type": "object", "properties": {}})
        assert n1 == "mcp__srv_a__do"
        assert n2 == "mcp__srv_b__do"
        assert registry.total_count == 2

    def test_unregister_server(self):
        """按服务器取消注册"""
        registry = MCPToolRegistry()
        registry.register("srv", "t1", "...", {"type": "object", "properties": {}})
        registry.register("srv", "t2", "...", {"type": "object", "properties": {}})
        assert registry.total_count == 2
        registry.unregister_server("srv")
        assert registry.total_count == 0

    def test_mark_unavailable(self):
        """工具标记为不可用"""
        registry = MCPToolRegistry()
        registry.register("srv", "t", "...", {"type": "object", "properties": {}})
        assert registry.available_count == 1
        registry.mark_server_unavailable("srv")
        assert registry.available_count == 0
        assert registry.total_count == 1  # 仍在注册表中

    def test_available_tools_filtered(self):
        """get_available_tools 仅返回可用工具"""
        registry = MCPToolRegistry()
        registry.register("srv", "t1", "...", {"type": "object", "properties": {}})
        registry.register("srv", "t2", "...", {"type": "object", "properties": {}})
        registry.mark_server_unavailable("srv")
        assert len(registry.get_available_tools()) == 0

    def test_approval_mode_default(self):
        """默认审批模式为 require_approval"""
        registry = MCPToolRegistry()
        full_name = registry.register("srv", "t", "...", {"type": "object", "properties": {}})
        assert registry.get_approval_mode(full_name) == "require_approval"

    def test_approval_mode_tool_override(self):
        """工具级审批覆盖"""
        registry = MCPToolRegistry()
        full_name = registry.register(
            "srv", "read_only", "只读工具",
            {"type": "object", "properties": {}},
            server_approval_mode="require_approval",
            tool_approval_overrides={"read_only": "auto_approve"},
        )
        assert registry.get_approval_mode(full_name) == "auto_approve"

    def test_approval_mode_unknown_tool(self):
        """未知工具返回默认 require_approval"""
        registry = MCPToolRegistry()
        assert registry.get_approval_mode("mcp__none__noop") == "require_approval"
