"""Prometheus 业务指标定义（供 main.py / agent / api 层引用）"""
from prometheus_client import Counter, Histogram, Gauge

# ==================== HTTP 层 (由 Instrumentator 自动采集，此处仅补自定义标签) ====================

# ==================== 对话/Chat 业务层 ====================
# 对话请求总数（按 mode + has_image 标签）
chat_requests = Counter(
    "z_agent_chat_requests_total",
    "对话请求总数",
    ["mode", "stream", "has_image"],
)

# 流式 SSE 事件数
sse_events = Counter(
    "z_agent_sse_events_total",
    "SSE 事件推送总数",
    ["event_type"],
)

# 活跃会话数（瞬时值）
active_sessions = Gauge(
    "z_agent_active_sessions",
    "当前活跃会话数",
)

# LLM token 消耗（近似，按字符估算）
llm_tokens = Counter(
    "z_agent_llm_tokens_total",
    "LLM token 消耗总数（字符估算）",
    ["model", "node"],
)

# LLM 调用延迟
llm_duration = Histogram(
    "z_agent_llm_duration_seconds",
    "LLM 调用耗时",
    ["model", "node"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 60],
)

# ReAct 迭代轮数分布
react_iterations = Histogram(
    "z_agent_react_iterations",
    "ReAct 循环迭代轮数",
    buckets=[1, 2, 3, 4, 5, 6],
)

# ReAct 工具调用次数（按工具名）
tool_calls = Counter(
    "z_agent_tool_calls_total",
    "工具调用总数",
    ["tool_name"],
)

# 答案验证结果
verification_results = Counter(
    "z_agent_verification_total",
    "答案验证结果",
    ["result"],
)

# ==================== 审批/HITL 层 ====================
# 审批请求创建数
approval_created = Counter(
    "z_agent_approval_created_total",
    "审批请求创建总数",
    ["tool_name", "server"],
)

# 审批决议数
approval_decisions = Counter(
    "z_agent_approval_decisions_total",
    "审批决议总数",
    ["decision"],
)

# 审批等待时长
approval_wait_duration = Histogram(
    "z_agent_approval_wait_seconds",
    "审批等待时长",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

# ==================== 检索层 ====================
# 检索次数
retrieval_count = Counter(
    "z_agent_retrieval_total",
    "检索调用总数",
    ["type"],
)

# 检索耗时
retrieval_duration = Histogram(
    "z_agent_retrieval_duration_seconds",
    "检索耗时",
    ["type"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

# ==================== MCP 层 ====================
# MCP 服务器连接状态
mcp_connection_status = Gauge(
    "z_agent_mcp_connection_status",
    "MCP 服务器连接状态 (1=已连接, 0=断开)",
    ["server"],
)

# MCP 工具调用数
mcp_tool_calls = Counter(
    "z_agent_mcp_tool_calls_total",
    "MCP 工具调用总数",
    ["server", "tool"],
)

# MCP 工具调用延迟
mcp_tool_duration = Histogram(
    "z_agent_mcp_tool_duration_seconds",
    "MCP 工具调用耗时",
    ["server", "tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
)

# ==================== 知识库层 ====================
# 文档上传次数
knowledge_uploads = Counter(
    "z_agent_knowledge_uploads_total",
    "知识库文档上传总数",
)

# 向量库文档总数（瞬时值）
vector_doc_count = Gauge(
    "z_agent_vector_doc_count",
    "向量库当前文档总数",
)
