"""Agent 工具定义"""
import logging
from contextvars import ContextVar
from langchain_core.tools import tool, StructuredTool

logger = logging.getLogger(__name__)

_rag_service: ContextVar = ContextVar("rag_service", default=None)


def set_rag_service(service):
    _rag_service.set(service)


def get_rag_service():
    svc = _rag_service.get()
    if svc is None:
        from core.rag_service import RagService
        svc = RagService()
        _rag_service.set(svc)
    return svc


def register_mcp_tools(mcp_registry) -> list:
    """将 MCP 注册表中的工具转为 LangChain StructuredTool 列表

    使用 MCP 工具的真实 inputSchema 作为 args_schema，
    让 LLM 看到正确的参数名（path, content 等），避免 kwargs 包裹问题。
    """

    def _make_func(srv: str, tname: str):
        def handler(**kwargs) -> str:
            return _call_mcp_tool(srv, tname, kwargs)
        return handler

    tools = []
    for meta in mcp_registry.get_available_tools():
        # 用 MCP 的 inputSchema 构建 args_schema，确保参数名一致
        from pydantic import create_model
        raw_schema = meta.input_schema
        props = raw_schema.get("properties", {})
        required = raw_schema.get("required", [])

        if props:
            # 动态创建 Pydantic 模型匹配真实参数
            fields = {}
            for prop_name, prop_info in props.items():
                prop_type = prop_info.get("type", "string")
                py_type = {"string": str, "number": float, "integer": int, "boolean": bool, "array": list, "object": dict}.get(prop_type, str)
                default = ... if prop_name in required else None
                fields[prop_name] = (py_type, default)
            args_model = create_model(f"{meta.full_name}_args", **fields)
        else:
            args_model = None

        t = StructuredTool.from_function(
            func=_make_func(meta.server_name, meta.original_name),
            name=meta.full_name,
            description=meta.description,
            args_schema=args_model,
        )
        tools.append(t)
        logger.debug("注册 MCP 工具到 agent: %s", meta.full_name)
    return tools


def _call_mcp_tool(server_name: str, tool_name: str, args: dict) -> str:
    """通过 MCP 管理器调用远程工具

    Args:
        server_name: MCP 服务器名
        tool_name: 原始工具名
        args: 工具参数字典

    Returns:
        工具执行结果文本
    """
    rag = get_rag_service()
    return rag.mcp_manager.call_tool(server_name, tool_name, args)


@tool
def search_knowledge(query: str, top_k: int = 6) -> str:
    """从知识库检索信息。回答内部问题时使用。

    返回格式含来源标记: [source | page:N | section:路径] 正文
    """
    rag = get_rag_service()
    try:
        # 优先使用层级检索（父子分块模式）
        ctxs, sources = rag.knowledge.hierarchical_retriever.retrieve(
            query, top_k=top_k, include_sources=True,
        )
        if ctxs:
            lines = []
            for i, (ctx, src) in enumerate(zip(ctxs, sources)):
                meta = f"{src.document or '?'}"
                if src.page:
                    meta += f" | 第{src.page}页"
                if src.section:
                    meta += f" | {src.section}"
                lines.append(f"[{meta}]\n{ctx}")
            return "\n\n---\n\n".join(lines)
    except Exception:
        pass

    # 降级：传统混合检索
    docs = rag.hybrid.search_as_documents(query, top_k=top_k)
    if not docs:
        return "未找到相关信息"
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source','')}] {d.page_content}" for d in docs
    )


