"""Windows 开发启动器：让 `--reload` 热重载与 MCP stdio 子进程共存。

问题：uvicorn 在 Windows 的 reload 工作进程里强制使用 SelectorEventLoop
（见 uvicorn/loops/asyncio.py：use_subprocess=True → SelectorEventLoop），
而 SelectorEventLoop 不支持 asyncio 子进程传输，会抛
`NotImplementedError`（MCP stdio 连接失败，空消息）。

修法：在 uvicorn 解析 loop factory 之前，把 uvicorn.loops.asyncio.asyncio_loop_factory
补丁为始终返回 ProactorEventLoop。uvicorn config.py 按字符串懒解析该符号，
故模块属性补丁会被正确拾取。正常（非 reload）模式原本就是 Proactor，不受影响。

用法（替代 uvicorn main:app --reload）：
    python run.py
可选环境变量：HOST / PORT / UVICORN_RELOAD（默认 reload 开启）
"""
import asyncio
import os
import sys

# 必须在 import/调用 uvicorn 之前补丁 loop factory
if sys.platform == "win32":
    import uvicorn.loops.asyncio as _aio_loop

    def _proactor_loop_factory(use_subprocess: bool = False):
        # 忽略 use_subprocess，Windows 下一律用 ProactorEventLoop 以支持子进程
        return asyncio.ProactorEventLoop

    _aio_loop.asyncio_loop_factory = _proactor_loop_factory


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    reload = os.getenv("UVICORN_RELOAD", "true").lower() in ("1", "true", "yes")

    if sys.platform == "win32":
        print(
            "[run.py] Windows 已补丁 uvicorn loop factory → ProactorEventLoop "
            "（reload 工作进程亦生效，MCP stdio 子进程可正常启动）",
            flush=True,
        )

    uvicorn.run("main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
