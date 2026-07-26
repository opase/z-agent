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
    from core import tracing
    from opentelemetry import metrics as otel_metrics

    # MCP 连接状态 Gauge（OTel Metrics API → OTLP → Phoenix /metrics）
    # 经 ObservableGauge 回调周期性读取实时连接状态，供 Prometheus 告警。
    meter = tracing.get_meter()

    def _mcp_status(_options):
        for server_name, ok in rag.mcp_manager.get_server_status().items():
            yield otel_metrics.Observation(1 if ok else 0, {"server": server_name})

    meter.create_observable_gauge(
        "z_agent_mcp_connection_status",
        callbacks=[_mcp_status],
        description="MCP 服务器连接状态 (1=已连接, 0=断开)",
    )

    # 启动
    logger.info("正在启动 MCP 服务器连接...")
    results = await rag.mcp_manager.start_all()
    connected = sum(1 for v in results.values() if v)
    logger.info("MCP 启动完成: %d/%d 连接成功", connected, len(results))

    yield  # 应用运行中

    # 关闭
    logger.info("正在关闭 MCP 服务器连接...")
    await rag.mcp_manager.shutdown_all()
    tracing.shutdown_tracing()  # 刷盘 in-flight span/metric


app = FastAPI(title="Zagent", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# OTel tracing 初始化（设 TracerProvider / MeterProvider + LangChain 自动 instrumentation）
from core import tracing
tracing.setup_tracing()
# 注：不启用 FastAPIInstrumentor。它产生的 HTTP server span（POST /chat/stream 等）不带
# OpenInference 的 kind/input/output，在 Phoenix 里是空行噪音，且会把 z_agent.chat 压成子 span、
# 让 trace 列表顶层显示空。去掉后 z_agent.chat 成为 trace 根 span，Phoenix 顶层直接显示
# 有值的 chat/LLM/tool 数据。若将来需要 HTTP 层埋点，再 instrument_app(app) 即可。


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
