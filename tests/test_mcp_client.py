"""MCP 客户端单元测试"""
import asyncio
import json
import pytest
from mcp_client.schema import normalize_tool_schema
from mcp_client.registry import MCPToolRegistry
from mcp_client.manager import MCPServerManager


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


class FakeTransport:
    """模拟 MCP 传输层——可配置 capabilities/tools/resources"""

    def __init__(self, capabilities=None, tools=None, resources=None, contents=None):
        self._capabilities = capabilities or {}
        self._tools = tools or []
        self._resources = resources or []
        self._contents = contents or []
        self.call_tool_invocations = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    def is_connected(self):
        return True

    async def initialize(self):
        return {"protocolVersion": "0.1", "capabilities": self._capabilities}

    async def list_tools(self):
        return self._tools

    async def call_tool(self, tool_name, arguments):
        self.call_tool_invocations.append((tool_name, arguments))
        return f"real:{tool_name}"

    async def list_resources(self):
        return self._resources

    async def read_resource(self, uri):
        return self._contents


def _make_manager(transport):
    """构造使用假传输层的 MCPServerManager"""
    registry = MCPToolRegistry()
    manager = MCPServerManager("nonexistent.yaml", registry)
    manager._create_transport = lambda cfg: transport
    return manager, registry


class TestResourceVirtualTools:
    """resources capability → 虚拟工具注册与分流测试"""

    ECHO_TOOL = {"name": "echo", "description": "回显", "inputSchema": {"type": "object", "properties": {}}}

    def test_register_virtual_tools_when_capability_present(self):
        """声明 resources capability 时注册两个虚拟工具"""
        transport = FakeTransport(capabilities={"resources": {}}, tools=[self.ECHO_TOOL])
        manager, registry = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        names = {m.full_name for m in registry.get_all_tools()}
        assert "mcp__srv__list_resources" in names
        assert "mcp__srv__read_resource" in names
        # read_resource 的 uri 为必填
        meta = registry.get_tool("mcp__srv__read_resource")
        assert meta.input_schema["required"] == ["uri"]

    def test_no_virtual_tools_without_capability(self):
        """未声明 resources capability 时不注册虚拟工具"""
        transport = FakeTransport(capabilities={"tools": {}}, tools=[self.ECHO_TOOL])
        manager, registry = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        names = {m.full_name for m in registry.get_all_tools()}
        assert names == {"mcp__srv__echo"}

    def test_skip_virtual_tool_on_name_collision(self):
        """服务器已有同名真实工具时跳过对应虚拟工具"""
        real_read = {"name": "read_resource", "description": "真实工具", "inputSchema": {"type": "object", "properties": {}}}
        transport = FakeTransport(capabilities={"resources": {}}, tools=[real_read])
        manager, registry = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        # read_resource 保留真实注册（描述来自服务器），list_resources 仍为虚拟工具
        assert registry.get_tool("mcp__srv__read_resource").description == "真实工具"
        assert registry.get_tool("mcp__srv__list_resources") is not None
        # 调用 read_resource 应透传给真实工具而非虚拟分流
        result = asyncio.run(manager.call_tool("srv", "read_resource", {"uri": "x"}))
        assert result == "real:read_resource"
        assert transport.call_tool_invocations == [("read_resource", {"uri": "x"})]

    def test_call_list_resources_formats_text(self):
        """list_resources 分流到 resources/list 并格式化输出"""
        transport = FakeTransport(
            capabilities={"resources": {}},
            resources=[
                {"uri": "file:///a.md", "name": "说明文档", "description": "项目说明", "mimeType": "text/markdown"},
                {"uri": "file:///b.bin"},
            ],
        )
        manager, _ = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        result = asyncio.run(manager.call_tool("srv", "list_resources", {}))
        assert "file:///a.md" in result
        assert "说明文档" in result
        assert "text/markdown" in result
        assert "file:///b.bin" in result
        # 未走 tools/call
        assert transport.call_tool_invocations == []

    def test_call_read_resource_returns_text(self):
        """read_resource 分流到 resources/read 并返回文本内容"""
        transport = FakeTransport(
            capabilities={"resources": {}},
            contents=[{"uri": "file:///a.md", "text": "hello resource"}],
        )
        manager, _ = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        result = asyncio.run(manager.call_tool("srv", "read_resource", {"uri": "file:///a.md"}))
        assert result == "hello resource"

    def test_read_resource_missing_uri(self):
        """read_resource 缺少 uri 返回错误提示"""
        transport = FakeTransport(capabilities={"resources": {}})
        manager, _ = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        result = asyncio.run(manager.call_tool("srv", "read_resource", {}))
        assert result.startswith("[错误]")
        assert "uri" in result

    def test_blob_content_placeholder(self):
        """二进制资源内容以占位符文本表示"""
        transport = FakeTransport(
            capabilities={"resources": {}},
            contents=[{"uri": "file:///img.png", "blob": "aGVsbG8=", "mimeType": "image/png"}],
        )
        manager, _ = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        result = asyncio.run(manager.call_tool("srv", "read_resource", {"uri": "file:///img.png"}))
        assert "二进制资源" in result
        assert "image/png" in result

    def test_empty_resources_and_contents(self):
        """空资源列表/空内容返回友好提示"""
        transport = FakeTransport(capabilities={"resources": {}})
        manager, _ = _make_manager(transport)
        assert asyncio.run(manager._connect_server({"name": "srv"}))
        assert "暂无 resources" in asyncio.run(manager.call_tool("srv", "list_resources", {}))
        assert "内容为空" in asyncio.run(manager.call_tool("srv", "read_resource", {"uri": "file:///x"}))
