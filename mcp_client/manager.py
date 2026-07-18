"""MCPServerManager — MCP 服务器生命周期管理"""

import asyncio
import logging
import os
from typing import Any

import yaml

from mcp_client.registry import MCPToolRegistry
from mcp_client.schema import normalize_tool_schema
from mcp_client.transport.stdio import StdioTransport
from mcp_client.transport.http import HttpTransport

logger = logging.getLogger(__name__)

# resource 虚拟工具定义：服务器声明 resources capability 时注册
# 让 LLM 通过普通工具调用发现/读取 MCP resources，复用现有审批与执行链路
_RESOURCE_VIRTUAL_TOOLS = (
    (
        "list_resources",
        "列出该 MCP 服务器暴露的 resources，返回 URI、名称、MIME 类型和描述",
        {"type": "object", "properties": {}},
    ),
    (
        "read_resource",
        "读取 MCP resource 内容。参数 uri 必须来自 list_resources 结果或用户明确提供的 resource URI",
        {
            "type": "object",
            "properties": {"uri": {"type": "string", "description": "要读取的 MCP resource URI"}},
            "required": ["uri"],
        },
    ),
)


def _format_resources(resources: list[dict]) -> str:
    """将 resources/list 结果格式化为 LLM 可读文本"""
    if not resources:
        return "该 MCP 服务器暂无 resources"
    lines = []
    for res in resources:
        if not isinstance(res, dict):
            continue
        line = f"- {res.get('uri', '')}"
        display = res.get("title") or res.get("name") or ""
        if display:
            line += f" — {display}"
        if res.get("description"):
            line += f"：{res['description']}"
        if res.get("mimeType"):
            line += f" [{res['mimeType']}]"
        lines.append(line)
    return "\n".join(lines) if lines else "该 MCP 服务器暂无 resources"


def _format_resource_contents(contents: list[dict]) -> str:
    """将 resources/read 结果格式化为文本，二进制内容以占位符表示"""
    if not contents:
        return "MCP resource 内容为空"
    parts = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        if item.get("text") is not None:
            parts.append(str(item["text"]))
        elif item.get("blob") is not None:
            parts.append(
                f"[二进制资源 {item.get('mimeType', '未知类型')}，"
                f"base64 长度 {len(str(item['blob']))}，不支持直接展示]"
            )
    return "\n".join(parts) if parts else "MCP resource 内容为空"


class MCPServerManager:
    """管理多个 MCP 服务器的连接、握手、工具发现和状态"""

    def __init__(self, config_path: str, registry: MCPToolRegistry):
        """
        Args:
            config_path: mcp_config.yaml 文件路径
            registry: MCP 工具注册表实例
        """
        self.config_path = config_path
        self.registry = registry
        self._servers: dict[str, dict[str, Any]] = {}
        self._initialized = False

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """获取所有服务器的状态摘要（供 /health 端点使用）"""
        result = {}
        for name, info in self._servers.items():
            transport = info["transport"]
            result[name] = {
                "status": "connected" if transport.is_connected() else "disconnected",
                "transport": info["config"]["transport"],
                "tools_count": len(self.registry.get_server_tools(name)),
            }
        return result

    async def start_all(self) -> dict[str, bool]:
        """启动所有已配置的 MCP 服务器连接

        Returns:
            {server_name: success} 字典
        """
        if self._initialized:
            logger.warning("MCPServerManager 已初始化，跳过重复启动")
            return {}

        config = self._load_config()
        servers_config = config.get("servers", [])
        settings = config.get("settings", {})

        if not servers_config:
            logger.info("MCP 配置为空，无服务器需要连接")
            self._initialized = True
            return {}

        results = {}
        for server_cfg in servers_config:
            name = server_cfg.get("name", "")
            if not name:
                logger.warning("跳过无名 MCP 服务器配置")
                continue

            success = await asyncio.wait_for(self._connect_server(server_cfg), timeout=15.0)
            results[name] = success

        self._initialized = True
        connected = sum(1 for v in results.values() if v)
        logger.info("MCP 启动完成: %d/%d 服务器连接成功", connected, len(results))
        return results

    async def _connect_server(self, server_cfg: dict) -> bool:
        """连接单个 MCP 服务器并注册其工具"""
        name = server_cfg["name"]
        transport_type = server_cfg.get("transport", "stdio")

        logger.info("连接 MCP 服务器 [%s] (%s)...", name, transport_type)

        # 创建传输实例
        transport = self._create_transport(server_cfg)
        if transport is None:
            return False

        # 连接
        if not await transport.connect():
            logger.error("MCP 服务器 [%s] 连接失败", name)
            self._servers[name] = {"config": server_cfg, "transport": transport, "error": "连接失败"}
            return False

        # 握手 + 工具发现
        try:
            init_result = await transport.initialize()
            capabilities = (init_result or {}).get("capabilities") or {}
            supports_resources = capabilities.get("resources") is not None
            tools = await transport.list_tools()

            approval_mode = server_cfg.get("approval_mode", "require_approval")
            tool_overrides = server_cfg.get("tool_overrides", {})

            # 注册工具
            registered = 0
            for tool in tools:
                raw_schema = tool.get("inputSchema", {})
                normalized = normalize_tool_schema(raw_schema)
                full_name = self.registry.register(
                    server_name=name,
                    tool_name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=normalized,
                    call_handler=lambda tn, a, s=name, t=transport: t.call_tool(tn, a),
                    server_approval_mode=approval_mode,
                    tool_approval_overrides=tool_overrides,
                )
                registered += 1

            # resources capability → 注册虚拟工具（与真实工具同名时跳过，避免覆盖）
            virtual_tools: set[str] = set()
            if supports_resources:
                real_names = {t.get("name") for t in tools}
                for vt_name, vt_desc, vt_schema in _RESOURCE_VIRTUAL_TOOLS:
                    if vt_name in real_names:
                        logger.warning(
                            "服务器 [%s] 已有同名真实工具 %s，跳过虚拟资源工具注册", name, vt_name,
                        )
                        continue
                    self.registry.register(
                        server_name=name,
                        tool_name=vt_name,
                        description=vt_desc,
                        input_schema=vt_schema,
                        server_approval_mode=approval_mode,
                        tool_approval_overrides=tool_overrides,
                    )
                    virtual_tools.add(vt_name)
                    registered += 1

            self._servers[name] = {
                "config": server_cfg,
                "transport": transport,
                "virtual_resource_tools": virtual_tools,
            }
            logger.info(
                "MCP 服务器 [%s] 连接成功: %d 个工具已注册（含 %d 个资源虚拟工具）",
                name, registered, len(virtual_tools),
            )
            return True

        except Exception as e:
            logger.error("MCP 服务器 [%s] 握手/工具发现失败: %s", name, e)
            await transport.disconnect()
            self._servers[name] = {"config": server_cfg, "transport": transport, "error": str(e)}
            return False

    def _create_transport(self, server_cfg: dict):
        """根据配置创建对应的传输实例"""
        transport_type = server_cfg.get("transport", "stdio")

        if transport_type == "stdio":
            command = server_cfg.get("command", [])
            if not command:
                logger.error("stdio 传输需要 command 配置")
                return None
            return StdioTransport(command)

        elif transport_type == "http":
            url = server_cfg.get("url", "")
            if not url:
                logger.error("HTTP 传输需要 url 配置")
                return None
            headers = server_cfg.get("headers", {})
            return HttpTransport(url, headers)

        else:
            logger.error("不支持的传输类型: %s", transport_type)
            return None

    def get_transport(self, server_name: str):
        """获取指定服务器的传输实例"""
        info = self._servers.get(server_name)
        if info:
            return info["transport"]
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """调用指定服务器的工具（含惰性重连）

        惰性重连逻辑（FR-007）:
        - 若传输层已断开，尝试一次重连
        - 重连成功 → 更新注册表状态 → 执行调用
        - 重连失败 → 标记工具不可用 → 返回错误
        """
        info = self._servers.get(server_name)
        if not info:
            return f"[错误] 未知 MCP 服务器: {server_name}"

        transport = info["transport"]

        # 惰性重连
        if not transport.is_connected():
            logger.info("服务器 [%s] 已断开，尝试惰性重连...", server_name)
            config = info["config"]
            try:
                if not await transport.connect():
                    self.registry.mark_server_unavailable(server_name)
                    return f"[错误] MCP 服务器 '{server_name}' 重连失败，工具不可用"
                # 重新握手
                await transport.initialize()
                self.registry.mark_server_available(server_name)
                logger.info("服务器 [%s] 惰性重连成功", server_name)
            except Exception as e:
                self.registry.mark_server_unavailable(server_name)
                return f"[错误] MCP 服务器 '{server_name}' 重连异常: {e}"

        try:
            import time
            from core import metrics
            t0 = time.time()
            if tool_name in info.get("virtual_resource_tools", ()):
                result = await self._call_resource_tool(transport, tool_name, arguments)
            else:
                result = await transport.call_tool(tool_name, arguments)
            metrics.mcp_tool_calls.labels(server=server_name, tool=tool_name).inc()
            metrics.mcp_tool_duration.labels(server=server_name, tool=tool_name).observe(time.time() - t0)
            return result
        except Exception as e:
            from core import metrics
            metrics.mcp_tool_calls.labels(server=server_name, tool=tool_name).inc()
            return f"[错误] MCP 工具调用失败: {e}"

    async def _call_resource_tool(self, transport, tool_name: str, arguments: dict) -> str:
        """执行 resource 虚拟工具（list_resources / read_resource）"""
        if tool_name == "list_resources":
            return _format_resources(await transport.list_resources())
        # read_resource
        uri = (arguments or {}).get("uri", "")
        if not uri:
            return "[错误] read_resource 缺少必填参数 uri"
        return _format_resource_contents(await transport.read_resource(uri))

    async def shutdown_all(self) -> None:
        """关闭所有 MCP 服务器连接"""
        logger.info("关闭所有 MCP 服务器连接 (共 %d 个)...", len(self._servers))
        for name, info in self._servers.items():
            transport = info.get("transport")
            if transport:
                try:
                    await transport.disconnect()
                    logger.info("服务器 [%s] 已关闭", name)
                except Exception as e:
                    logger.warning("关闭服务器 [%s] 时出错: %s", name, e)
        self._servers.clear()
        self._initialized = False

    def _load_config(self) -> dict:
        """加载并解析 MCP 配置文件"""
        if not os.path.exists(self.config_path):
            logger.warning("MCP 配置文件不存在: %s，使用空配置", self.config_path)
            return {"servers": [], "settings": {}}

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {"servers": [], "settings": {}}
