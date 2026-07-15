"""MCP 工具注册表 — 管理已发现工具的命名空间和审批策略"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPToolMetadata:
    """MCP 工具的元数据"""
    full_name: str          # 命名空间全名: mcp__{server}__{tool}
    server_name: str        # 来源 MCP 服务器
    original_name: str      # 服务器端原始工具名
    description: str        # 工具描述
    input_schema: dict      # 经裁剪的 JSON Schema
    approval_mode: str      # "require_approval" | "auto_approve"
    status: str = "available"  # "available" | "unavailable"
    call_handler: Callable | None = None  # 实际调用 MCP 服务器的函数


class MCPToolRegistry:
    """MCP 工具注册表

    管理所有从 MCP 服务器发现的工具，提供：
    - 命名空间注册（mcp__{server}__{tool} 格式）
    - 工具状态管理（available / unavailable）
    - 审批策略查询（服务器级默认 + 工具级覆盖）
    """

    def __init__(self):
        self._tools: dict[str, MCPToolMetadata] = {}
        self._server_approval_modes: dict[str, str] = {}
        self._tool_approval_overrides: dict[str, dict[str, str]] = {}

    def register(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict,
        call_handler: Callable | None = None,
        server_approval_mode: str = "require_approval",
        tool_approval_overrides: dict[str, str] | None = None,
    ) -> str:
        """注册一个 MCP 工具

        Args:
            server_name: MCP 服务器名称
            tool_name: 原始工具名
            description: 工具描述文本
            input_schema: 经裁剪的参数 Schema
            call_handler: 实际调用 MCP 服务器的异步函数
            server_approval_mode: 服务器级别的默认审批策略
            tool_approval_overrides: 工具级别的审批策略覆盖 {tool_name: mode}

        Returns:
            工具的命名空间全名 (mcp__{server}__{tool})
        """
        full_name = f"mcp__{server_name}__{tool_name}"

        if full_name in self._tools:
            logger.warning("工具 %s 已注册，将被覆盖", full_name)

        # 计算最终审批模式：工具级覆盖 > 服务器级
        approval_mode = server_approval_mode
        if tool_approval_overrides and tool_name in tool_approval_overrides:
            approval_mode = tool_approval_overrides[tool_name]

        metadata = MCPToolMetadata(
            full_name=full_name,
            server_name=server_name,
            original_name=tool_name,
            description=description,
            input_schema=input_schema,
            approval_mode=approval_mode,
            status="available",
            call_handler=call_handler,
        )

        self._tools[full_name] = metadata
        logger.info("MCP 工具已注册: %s (审批: %s)", full_name, approval_mode)
        return full_name

    def unregister_server(self, server_name: str) -> int:
        """取消注册指定服务器的所有工具

        Returns:
            移除的工具数量
        """
        removed = 0
        for full_name in list(self._tools.keys()):
            if self._tools[full_name].server_name == server_name:
                del self._tools[full_name]
                removed += 1
        logger.info("已移除服务器 %s 的 %d 个工具", server_name, removed)
        return removed

    def mark_server_unavailable(self, server_name: str) -> None:
        """将指定服务器的所有工具标记为不可用"""
        for meta in self._tools.values():
            if meta.server_name == server_name:
                meta.status = "unavailable"
        logger.warning("服务器 %s 的工具已标记为不可用", server_name)

    def mark_server_available(self, server_name: str) -> None:
        """将指定服务器的所有工具恢复为可用"""
        for meta in self._tools.values():
            if meta.server_name == server_name:
                meta.status = "available"
        logger.info("服务器 %s 的工具已恢复为可用", server_name)

    def get_tool(self, full_name: str) -> MCPToolMetadata | None:
        """获取指定工具的元数据"""
        return self._tools.get(full_name)

    def get_available_tools(self) -> list[MCPToolMetadata]:
        """获取所有可用状态的工具"""
        return [m for m in self._tools.values() if m.status == "available"]

    def get_all_tools(self) -> list[MCPToolMetadata]:
        """获取所有工具（含不可用）"""
        return list(self._tools.values())

    def get_approval_mode(self, full_name: str) -> str:
        """查询指定工具的审批策略

        查询优先级: 工具级覆盖 > 服务器级默认 > 全局默认 require_approval

        Returns:
            "require_approval" 或 "auto_approve"
        """
        tool = self._tools.get(full_name)
        if tool:
            return tool.approval_mode
        return "require_approval"

    def get_server_tools(self, server_name: str) -> list[MCPToolMetadata]:
        """获取指定服务器的所有工具"""
        return [m for m in self._tools.values() if m.server_name == server_name]

    @property
    def total_count(self) -> int:
        return len(self._tools)

    @property
    def available_count(self) -> int:
        return sum(1 for m in self._tools.values() if m.status == "available")
