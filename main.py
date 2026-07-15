"""
Zagent智能问答助手 - 入口
"""
import uuid
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from core.rag_service import RagService
from core.knowledge_service import KnowledgeService
from core.session_service import SessionManager
from core.session_store import SQLiteStore
from api import chat, knowledge, user, approval, state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 启动校验 ──
from config import settings as config
if not config.dashscope_api_key:
    raise RuntimeError(
        "DASHSCOPE_API_KEY 未设置。请在环境变量中设置：\n"
        "  export DASHSCOPE_API_KEY=sk-xxxx\n"
        "或修改 config/settings.py 中的 dashscope_api_key"
    )

# ── 初始化核心服务 ──
rag = RagService()
rag.sync_bm25()
knowledge_svc = KnowledgeService()
session_mgr = SessionManager(
    llm=rag.llm,
    store=SQLiteStore(),
    light_llm=rag.light_llm,
)


# ── Lifespan 事件处理（替代已弃用的 on_event）──
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期：启动时连接 MCP，关闭时清理子进程"""
    from core import metrics

    # 启动
    logger.info("正在启动 MCP 服务器连接...")
    results = await rag.mcp_manager.start_all()
    for server_name, ok in results.items():
        metrics.mcp_connection_status.labels(server=server_name).set(1 if ok else 0)
    connected = sum(1 for v in results.values() if v)
    logger.info("MCP 启动完成: %d/%d 连接成功", connected, len(results))

    yield  # 应用运行中

    # 关闭
    logger.info("正在关闭 MCP 服务器连接...")
    for server_name in rag.mcp_manager.server_names:
        metrics.mcp_connection_status.labels(server=server_name).set(0)
    await rag.mcp_manager.shutdown_all()


app = FastAPI(title="Zagent", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# Prometheus 指标暴露
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
).instrument(app).expose(app)


# 请求追踪中间件
@app.middleware("http")
async def request_tracing(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id
    start = time.time()
    logger.info("[%s] %s %s", request_id, request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Request-ID"] = request_id
    logger.info("[%s] %s %s → %d (%.2fs)", request_id, request.method, request.url.path, response.status_code, elapsed)
    return response


# 挂载服务到 app.state
app.state.rag = rag
app.state.knowledge = knowledge_svc
app.state.sessions = session_mgr

# 注册路由
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(user.router)
app.include_router(approval.router)
app.include_router(state.router)


@app.get("/")
async def root():
    return {"service": "Zagent", "version": "2.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "bm25_docs": rag.bm25.doc_count,
        "active_sessions": len(session_mgr.sessions),
        "mcp_servers": rag.mcp_manager.get_server_status(),
    }
