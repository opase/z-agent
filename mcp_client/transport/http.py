"""HttpTransport — 通过 HTTP + SSE 进行 MCP JSON-RPC 通信"""

import asyncio
import json
import logging
import uuid
from typing import Any

import aiohttp

from mcp_client.transport import TransportInterface

logger = logging.getLogger(__name__)


class HttpTransport(TransportInterface):
    """基于 aiohttp 的 MCP HTTP+SSE 传输"""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        """
        Args:
            url: MCP 服务器 URL
            headers: 自定义 HTTP 请求头（如 Authorization）
        """
        self.url = url.rstrip("/")
        self.headers = headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._connected = False
        self._request_id = 0
        self._sse_url: str | None = None

    async def connect(self) -> bool:
        """建立 HTTP 连接并发现 SSE 端点"""
        try:
            self._session = aiohttp.ClientSession(headers={
                "Content-Type": "application/json",
                **self.headers,
            })
            # 尝试获取 SSE 端点（如果服务器支持）
            try:
                async with self._session.get(f"{self.url}/sse", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.split("\n"):
                            if line.startswith("data: "):
                                data = json.loads(line[6:])
                                self._sse_url = data.get("uri") or data.get("url")
                                break
            except Exception:
                # SSE 端点不是必须的
                self._sse_url = None

            self._connected = True
            logger.info("HttpTransport 连接成功: %s", self.url)
            return True
        except aiohttp.ClientError as e:
            logger.error("HTTP MCP 连接失败: %s — %s", self.url, e)
            self._connected = False
            return False
        except Exception as e:
            logger.error("HTTP MCP 连接失败: %s — %s", self.url, e)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """关闭 HTTP 会话"""
        self._connected = False
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("HttpTransport 已断开: %s", self.url)

    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP initialize 握手"""
        result = await self._send_request("initialize", {
            "protocolVersion": "0.1",
            "capabilities": {},
            "clientInfo": {"name": "z-agent", "version": "1.0"},
        })
        # 发送 initialized 通知
        await self._send_notification("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取工具列表"""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回文本结果"""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts) if text_parts else json.dumps(result, ensure_ascii=False)

    async def _send_request(self, method: str, params: dict) -> dict[str, Any]:
        """发送 JSON-RPC HTTP 请求"""
        if not self.is_connected():
            raise ConnectionError("MCP HTTP 服务器未连接")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": str(self._request_id),
            "method": method,
            "params": params,
        }

        try:
            async with self._session.post(
                self.url,
                json=request,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise ConnectionError(f"MCP HTTP 错误 {resp.status}: {text[:200]}")

                response = await resp.json()
                if "error" in response:
                    raise RuntimeError(response["error"].get("message", "MCP 错误"))
                return response.get("result", {})
        except aiohttp.ClientError as e:
            self._connected = False
            raise ConnectionError(f"MCP HTTP 请求失败: {e}")

    async def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（无响应）"""
        if not self.is_connected():
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            async with self._session.post(
                self.url,
                json=notification,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                pass  # 通知无需处理响应
        except Exception as e:
            logger.warning("发送 MCP 通知失败: %s", e)
