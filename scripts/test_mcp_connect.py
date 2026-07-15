"""MCP 连接测试 — 验证与 filesystem 服务器的协议握手"""
import asyncio
import logging
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")

# 配置日志以便调试
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from mcp_client.transport.stdio import StdioTransport
from mcp_client.schema import normalize_tool_schema


def ok(msg):
    print(f"[OK] {msg}")

def fail(msg):
    print(f"[FAIL] {msg}")


async def main():
    command = ["npx.cmd", "-y", "@modelcontextprotocol/server-filesystem", "."]
    print(f"Starting MCP server: {' '.join(command)}")

    transport = StdioTransport(command)

    print("Connecting...")
    connected = await transport.connect()
    if not connected:
        fail("Connection failed")
        return
    ok("Connected")

    print("Handshaking (initialize)...")
    try:
        caps = await transport.initialize()
        ok(f"Handshake success: protocol={caps.get('protocolVersion', '?')}")
    except Exception as e:
        fail(f"Handshake failed: {e}")
        await transport.disconnect()
        return

    print("Listing tools (tools/list)...")
    try:
        tools = await transport.list_tools()
        ok(f"Found {len(tools)} tools:")
        for tool in tools:
            name = tool.get("name", "?")
            desc = tool.get("description", "")[:80]
            schema = tool.get("inputSchema", {})
            normalized = normalize_tool_schema(schema)
            print(f"  - {name}: {desc}")
            print(f"    Raw schema keys: {list(schema.keys())}")
            print(f"    Normalized keys: {list(normalized.keys())}")
    except Exception as e:
        fail(f"List tools failed: {e}")
        import traceback
        traceback.print_exc()
        await transport.disconnect()
        return

    # Test calling a read-only tool
    if tools:
        safe_tool = None
        for tool in tools:
            if tool.get("name") in ("list_directory", "read_file", "list_allowed_directories"):
                safe_tool = tool
                break
        if safe_tool is None:
            safe_tool = tools[0]

        tool_name = safe_tool["name"]
        print(f"Testing tool call: {tool_name}...")
        try:
            if tool_name == "list_allowed_directories":
                result = await transport.call_tool(tool_name, {})
            elif tool_name == "list_directory":
                result = await transport.call_tool(tool_name, {"path": "."})
            elif tool_name == "read_file":
                result = await transport.call_tool(tool_name, {"path": ".\\README.md"})
            else:
                result = await transport.call_tool(tool_name, {})

            display = result[:500] if len(result) > 500 else result
            ok(f"Call success ({len(result)} chars):\n{display}")
            if len(result) > 500:
                print(f"... (total {len(result)} chars)")
        except Exception as e:
            fail(f"Call failed: {e}")
            import traceback
            traceback.print_exc()

    await transport.disconnect()
    ok("Test complete")


if __name__ == "__main__":
    asyncio.run(main())
