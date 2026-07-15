"""
服务容器 — 按子系统分组构建组件，替代 RagService.__init__ 中的内联构造

每个容器负责：
- 一个子系统的组件实例化与装配
- 可独立单元测试
- 新增子系统不再往 RagService 里堆代码
"""
from langchain_community.chat_models import ChatTongyi
from config import settings as config

from retrieval.embedding import EmbeddingService
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever, HybridRetriever
from retrieval.reranker import Reranker

from mcp_client.registry import MCPToolRegistry
from mcp_client.manager import MCPServerManager
from agent.approval import ApprovalManager
from core.approval_store import ApprovalStore

from agent.vision import VisionAnalyzer
from memory.user_profile import UserProfileExtractor


class RetrievalServices:
    """检索子系统：embedding + BM25 + vector + hybrid + reranker"""

    def __init__(self):
        self.embedding = EmbeddingService()
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(self.embedding)
        self.hybrid = HybridRetriever(self.vector, self.bm25)
        self.reranker = Reranker()


class LLMProvider:
    """LLM 实例集合 — 按用途区分模型与参数

    - llm:         流式对话（chat_stream 最终输出）
    - light_llm:   轻量任务（意图分类、验证、压缩摘要）
    - react_llm:   工具调用（非流式，LangChain tool calling 要求）
    """

    def __init__(self):
        self.llm = ChatTongyi(
            model=config.chat_model,
            dashscope_api_key=config.dashscope_api_key,
            streaming=True,
        )
        self.light_llm = ChatTongyi(
            model=config.classifier_model,
            dashscope_api_key=config.dashscope_api_key,
            temperature=0,
        )
        self.react_llm = ChatTongyi(
            model=config.chat_model,
            dashscope_api_key=config.dashscope_api_key,
            streaming=False,  # tool calling 必须非流式
        )


class MCPServices:
    """MCP 子系统：工具注册表 + 连接管理 + 审批存储与管理"""

    def __init__(self):
        self.registry = MCPToolRegistry()
        self.manager = MCPServerManager(config.mcp_config_path, self.registry)
        self.approval_store = ApprovalStore()
        self.approval_mgr = ApprovalManager()
        self.approval_mgr.set_store(self.approval_store)


class VisionServices:
    """视觉子系统：多模态分析"""

    def __init__(self):
        self.vision_analyzer = VisionAnalyzer()


class MemoryServices:
    """记忆子系统：用户画像提取"""

    def __init__(self):
        self.profile_extractor = UserProfileExtractor()
