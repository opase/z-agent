"""MCP 客户端模块 — Model Context Protocol 集成层

提供 MCP 服务器连接管理、工具注册、Schema 规范化功能。
"""
from mcp_client.manager import MCPServerManager
from mcp_client.registry import MCPToolRegistry
from mcp_client.schema import normalize_tool_schema

__all__ = ["MCPServerManager", "MCPToolRegistry", "normalize_tool_schema"]
