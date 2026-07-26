"""
统一配置文件
所有配置项集中管理，支持 .env 文件与环境变量覆盖
"""
import os


def _load_dotenv():
    """零依赖 .env 加载器：读取项目根目录 .env，仅在对应环境变量未设置时注入。

    - 行格式 KEY=VALUE；`#` 开头或空行忽略；值两侧引号自动去除。
    - 已存在的 shell 环境变量优先，不被 .env 覆盖。
    - 空值（KEY=）忽略，保留代码内默认值。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

# ==================== 服务配置 ====================
HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8080"))

# ==================== DashScope API ====================
# 密钥从 .env（或 shell 环境变量）读取，请勿硬编码提交到仓库。
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

# ==================== 模型配置 ====================
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
chat_model = os.getenv("CHAT_MODEL", "glm-5.1")            # ReAct 工具调用 / 流式回答
rerank_model = os.getenv("RERANK_MODEL", "gte-rerank-v2")
classifier_model = os.getenv("CLASSIFIER_MODEL", "qwen-turbo")  # 意图分类、验证、画像提取等轻量任务
vision_model = os.getenv("VISION_MODEL", "qwen3-vl")      # 多模态视觉理解

# ==================== 图片上传限制 ====================
max_images_per_message = 3
max_image_size_mb = 10

# ==================== 文档分块 ====================
chunk_size = 500
chunk_overlap = 100
separators = ["\n\n", "\n", "。", "？", "！", "；", "，", "、", " ", ""]
max_split_char_number = 1000

# ==================== 文档解析器（可插拔）====================
# "auto": MinerU 可用则用，不可用降级 PlainTextParser
# "plain": 始终纯文本（零额外依赖）
# "mineru": 强制 MinerU，不可用时启动报错
document_parser = os.getenv("DOCUMENT_PARSER", "auto")
# MinerU 子模式（仅 DOCUMENT_PARSER=auto/mineru 时生效）
# "auto" / "cli" / "api-flash" / "api-precision"
mineru_method = os.getenv("MINERU_METHOD", "auto")
# 云端精准模式 token
mineru_api_token = os.getenv("MINERU_API_TOKEN", "")

# ==================== 文档入库流水线 ====================
ingestion_cleaning_enabled = os.getenv("INGESTION_CLEANING", "true").lower() in ("1", "true", "yes")
ingestion_dedup_enabled = os.getenv("INGESTION_DEDUP", "true").lower() in ("1", "true", "yes")
ingestion_dedup_threshold = float(os.getenv("INGESTION_DEDUP_THRESHOLD", "0.92"))

# ==================== 向量数据库 ====================
collection_name = "rag"
persist_directory = ".chroma_db"

# ==================== 检索配置 ====================
bm25_top_k = 10
vector_top_k = 10
hybrid_top_k = 10
rrf_k = 60
similarity_num = 6
rerank_top_k = 6

# ==================== MCP 配置 ====================
mcp_config_path = os.getenv("MCP_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_config.yaml"))
approval_timeout_minutes = int(os.getenv("APPROVAL_TIMEOUT_MINUTES", "5"))
approval_max_tools_per_server = int(os.getenv("APPROVAL_MAX_TOOLS_PER_SERVER", "20"))
max_mcp_servers = int(os.getenv("MAX_MCP_SERVERS", "10"))

# ==================== SSE 事件格式 ====================
sse_structured_events = os.getenv("SSE_STRUCTURED_EVENTS", "true").lower() in ("1", "true", "yes")

# ==================== OpenTelemetry / Phoenix 追踪 ====================
# Phoenix 为主（Trace 驱动），Prometheus 为辅（聚合告警）。
# Span 经 OTLP/HTTP 发往 Phoenix。
otel_enabled = os.getenv("OTEL_ENABLED", "true").lower() in ("1", "true", "yes")
otel_service_name = os.getenv("OTEL_SERVICE_NAME", "z-agent")
otel_environment = os.getenv("OTEL_ENVIRONMENT", "development")
# OTLP/HTTP 端点必须含路径：traces 用 /v1/traces，metrics 用 /v1/metrics
phoenix_traces_endpoint = os.getenv("PHOENIX_TRACES_ENDPOINT", "http://localhost:6006/v1/traces")
phoenix_metrics_endpoint = os.getenv("PHOENIX_METRICS_ENDPOINT", "http://localhost:6006/v1/metrics")
phoenix_api_key = os.getenv("PHOENIX_API_KEY", "")  # Phoenix 云端/鉴权时填，留空不发 Authorization 头
# 2 个瞬时 Gauge（active_sessions / mcp_connection_status）是否经 OTLP metrics 发送。
# 默认关闭：旧版 Phoenix 仅支持 /v1/traces，不摄取 OTLP metrics（返回 405），
# 会导致 PeriodicExportingMetricReader 每 10s 刷一条 ERROR 日志。
# 启用条件（满足任一）：① 确认你的 Phoenix 版本支持 OTLP HTTP metrics 摄取；
#                      ② 改用 OTel Collector 作为 metrics 后端（phoenix_metrics_endpoint 指向 Collector）。
# 关闭时 Gauge 退化为 no-op（get_meter() 返回默认空 meter），traces 不受影响。
otel_metrics_enabled = os.getenv("OTEL_METRICS_ENABLED", "false").lower() in ("1", "true", "yes")
otel_debug_console = os.getenv("OTEL_DEBUG_CONSOLE", "false").lower() in ("1", "true", "yes")

# ==================== 评估 / 消融实验（写入 span 属性，供 Phoenix 聚合对比）====================
# 语义成功率评估：每轮对话额外调一次 light_llm 做 LLM-as-judge（pass/score/reason）。
# 默认关闭——开启后每轮多一次 LLM 调用（有成本/延迟），仅在做成功率统计时打开。
eval_semantic_enabled = os.getenv("EVAL_SEMANTIC_ENABLED", "false").lower() in ("1", "true", "yes")
# 消融实验标签：改实验参数时改这些值，Phoenix 按属性 group-by 即可对比不同方案。
# - experiment_id：本次实验标识（如 "prompt-v2-run3"）
# - prompt_version：系统 prompt 版本
# - retrieval_scheme：检索方案（hybrid / vector / bm25）
# - tool_desc_version：工具描述版本
experiment_id = os.getenv("EXPERIMENT_ID", "")
prompt_version = os.getenv("PROMPT_VERSION", "default")
retrieval_scheme = os.getenv("RETRIEVAL_SCHEME", "hybrid")
tool_desc_version = os.getenv("TOOL_DESC_VERSION", "default")

# ==================== Agent 执行限制 ====================
max_react_iterations = int(os.getenv("MAX_REACT_ITERATIONS", "5"))       # ReAct 最大迭代轮数
max_task_iterations = int(os.getenv("MAX_TASK_ITERATIONS", "5"))         # Plan 模式单任务最大迭代轮数
max_retries_per_step = int(os.getenv("MAX_RETRIES_PER_STEP", "2"))      # Multi-Agent Reviewer 最大重试次数
replan_threshold = float(os.getenv("REPLAN_THRESHOLD", "0.5"))          # 失败后触发重规划的进度阈值

# ==================== 记忆配置 ====================
memory_window_size = 10
session_timeout_hours = 24

# ==================== 压缩配置（Phase 3: Map-Reduce）====================
compression_chunk_size = 5          # Map 阶段每块消息数
compression_max_summary_chars = 800 # 摘要硬截断上限（安全网）
compression_retained_rounds = 3     # 压缩时保留的最近对话轮数（每轮=user+assistant）
compression_reduce_max_chars = 300  # Reduce 阶段合并输出上限
compression_map_max_chars = 200     # Map 阶段单块输出上限

# ==================== Token 预算与上下文压缩 ====================
# 支持环境变量覆盖，便于测试时临时调小窗口触发压缩（如 MODEL_CONTEXT_WINDOW=2000）
model_context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))  # 模型上下文窗口（token）
compression_trigger_ratio = float(os.getenv("COMPRESSION_TRIGGER_RATIO", "0.9"))  # 占用达该比例触发压缩
react_retain_recent_rounds = int(os.getenv("REACT_RETAIN_RECENT_ROUNDS", "3"))    # ReAct 压缩保留的最近工具轮数
short_term_memory_ratio = float(os.getenv("SHORT_TERM_MEMORY_RATIO", "0.45"))     # 短期记忆预算 = 窗口 × 该比例
# 短期记忆压缩触发阈值（token）：预算 × 触发率
short_term_memory_trigger_tokens = int(
    model_context_window * short_term_memory_ratio * compression_trigger_ratio
)

# ==================== 数据路径 ====================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
md5_path = os.path.join(DATA_DIR, "md5.text")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
MEMORY_DIR = os.path.join(DATA_DIR, "user_memory")
