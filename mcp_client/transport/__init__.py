"""MCP 传输层抽象"""

from abc import ABC, abstractmethod
from typing import Any


class TransportInterface(ABC):
    """MCP 传输层抽象基类——定义 stdio/HTTP 传输的统一接口"""

    @abstractmethod
    async def connect(self) -> bool:
        """建立与 MCP 服务器的连接，返回是否成功"""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """断开与 MCP 服务器的连接并清理资源"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """返回当前连接状态"""
        ...

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 服务器的指定工具，返回工具执行结果文本"""
        ...

    @abstractmethod
    async def initialize(self) -> dict[str, Any]:
        """执行 MCP 协议握手（initialize → initialized），返回服务器 capabilities"""
        ...

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """获取服务器提供的工具列表，每个工具含 name/description/inputSchema"""
        ...

    @abstractmethod
    async def list_resources(self) -> list[dict[str, Any]]:
        """获取服务器暴露的资源列表，每个资源含 uri/name/description/mimeType"""
        ...

    @abstractmethod
    async def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """读取指定 URI 的资源内容，返回 contents 列表（text 或 blob 项）"""
        ...
