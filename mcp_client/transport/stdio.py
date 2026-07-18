"""StdioTransport — 通过子进程 stdin/stdout 进行 MCP JSON-RPC 通信"""

import asyncio
import json
import logging
import uuid
from typing import Any

from mcp_client.transport import TransportInterface

logger = logging.getLogger(__name__)

# 心跳间隔（秒）
HEARTBEAT_INTERVAL = 30
# 子进程关闭等待超时（秒）
PROCESS_TERMINATE_TIMEOUT = 3
# stdout 单行缓冲上限：MCP JSON-RPC 每条消息是一整行，默认 64 KiB 太小，
# 读多个大文件时单行会溢出（LimitOverrunError），故调大到 16 MiB。
MAX_LINE_BYTES = 16 * 1024 * 1024


class StdioTransport(TransportInterface):
    """基于 asyncio 子进程的 MCP stdio 传输"""

    def __init__(self, command: list[str]):
        """
        Args:
            command: 可执行文件及参数列表，如 ["python", "mcp_server.py"]
        """
        self.command = command
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False
        self._request_id = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        """启动子进程并完成 MCP 协议握手"""
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_LINE_BYTES,  # 提高单行缓冲上限，支持大响应（如批量读文件）
            )
            # 短暂等待，检查进程是否立即退出（npx 首次下载包可能需要数秒）
            await asyncio.sleep(2.0)
            if self._process.returncode is not None:
                stderr_data = await self._process.stderr.read()
                stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
                logger.error(
                    "MCP 子进程立即退出 (exit=%d): cmd=%s stderr=%s",
                    self._process.returncode, self.command[0], stderr_text[:500],
                )
                self._connected = False
                return False

            self._connected = True
            self._read_task = asyncio.create_task(self._read_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("StdioTransport 子进程已启动: %s (pid=%d)", self.command[0], self._process.pid)
            return True
        except FileNotFoundError:
            logger.error("无法启动 MCP 子进程: 文件未找到 '%s'", self.command[0])
            self._connected = False
            return False
        except OSError as e:
            logger.error("启动 MCP 子进程 OS 错误: %s (cmd=%s)", e, self.command[0])
            self._connected = False
            return False
        except Exception as e:
            logger.error("启动 MCP 子进程失败: %s (type=%s)", e, type(e).__name__)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """终止子进程并清理资源"""
        self._connected = False

        # 停止心跳和读取循环
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._read_task:
            self._read_task.cancel()
            self._read_task = None

        # 取消所有等待中的请求
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("MCP 连接已断开"))
        self._pending.clear()

        # 终止子进程
        if self._process:
            try:
                if self._process.returncode is None:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=PROCESS_TERMINATE_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.warning("子进程 %d 未响应 SIGTERM，发送 SIGKILL", self._process.pid)
                        self._process.kill()
                        await self._process.wait()
                    # Windows: 等待管道完全关闭
                    await asyncio.sleep(0.1)
                else:
                    # 进程已退出，确保 wait() 被调用以清理管道
                    await self._process.wait()
                logger.info("子进程 %d 已关闭 (exit=%d)", self._process.pid, self._process.returncode)
            except ProcessLookupError:
                pass  # 进程已退出
            except Exception as e:
                logger.warning("关闭子进程时出错: %s", e)
            self._process = None

    def is_connected(self) -> bool:
        """检查连接状态（子进程存活且正常）"""
        if not self._connected:
            return False
        if self._process is None:
            return False
        if self._process.returncode is not None:
            # 子进程已退出
            self._connected = False
            return False
        return True

    async def initialize(self) -> dict[str, Any]:
        """执行 MCP initialize 握手"""
        result = await self._send_request("initialize", {
            "protocolVersion": "0.1",
            "capabilities": {},
            "clientInfo": {"name": "z-agent", "version": "1.0"},
        })
        # 发送 initialized 通知
        self._send_notification("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取工具列表"""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def list_resources(self) -> list[dict[str, Any]]:
        """获取资源列表"""
        result = await self._send_request("resources/list", {})
        return result.get("resources", [])

    async def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """读取指定 URI 的资源内容"""
        result = await self._send_request("resources/read", {"uri": uri})
        return result.get("contents", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回文本结果"""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # 提取 content 中的 text 部分
        content = result.get("content", [])
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts) if text_parts else json.dumps(result, ensure_ascii=False)

    async def _send_request(self, method: str, params: dict) -> dict[str, Any]:
        """发送 JSON-RPC 请求并等待响应"""
        if not self.is_connected():
            raise ConnectionError("MCP 服务器未连接")

        self._request_id += 1
        req_id = str(self._request_id)
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        raw = json.dumps(request, ensure_ascii=False) + "\n"
        try:
            self._process.stdin.write(raw.encode("utf-8"))
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            self._connected = False
            self._pending.pop(req_id, None)
            raise ConnectionError(f"写入 MCP 进程失败: {e}")

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP 请求超时: {method}")

    def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（无响应）"""
        if not self.is_connected():
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        raw = json.dumps(notification, ensure_ascii=False) + "\n"
        try:
            self._process.stdin.write(raw.encode("utf-8"))
            # 不在 async 上下文中不能 await drain — 依赖缓冲区
        except Exception as e:
            logger.warning("发送 MCP 通知失败: %s", e)

    async def _read_loop(self) -> None:
        """持续读取子进程 stdout 的 JSON-RPC 响应"""
        buffer = ""
        while self._connected and self._process and self._process.stdout:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    # EOF — 进程退出
                    logger.warning("MCP 子进程 stdout 已关闭 (pid=%d)", self._process.pid)
                    self._connected = False
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                response = json.loads(line_str)
                req_id = response.get("id")
                if req_id and str(req_id) in self._pending:
                    future = self._pending.pop(str(req_id))
                    if not future.done():
                        if "error" in response:
                            future.set_exception(
                                RuntimeError(response["error"].get("message", "MCP 错误"))
                            )
                        else:
                            future.set_result(response.get("result", {}))
            except asyncio.CancelledError:
                break
            except json.JSONDecodeError as e:
                logger.warning("MCP JSON 解析失败: %s", e)
            except Exception as e:
                if self._connected:
                    logger.error("MCP 读取循环异常: %s", e)

        # 读取循环退出后标记断开
        if self._connected:
            self._connected = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("MCP 连接意外断开"))
            self._pending.clear()

    async def _heartbeat_loop(self) -> None:
        """定期心跳检测连接活性（通过 tools/list 轻量请求）"""
        while self._connected:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if not self._connected:
                break
            try:
                await self._send_request("tools/list", {})
                logger.debug("MCP 心跳正常: %s", self.command[0])
            except Exception as e:
                logger.warning("MCP 心跳失败: %s", e)
                self._connected = False
                break
