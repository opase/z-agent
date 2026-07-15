"""测试 Agent 图结构"""
import pytest


@pytest.fixture(scope="module")
def graph():
    """构建图（不依赖 LLM 进行结构测试）"""
    try:
        from core.rag_service import RagService
        rag = RagService()
        return rag.agent_graph
    except Exception:
        pytest.skip("无法构建 Agent 图（可能缺少 API key）")


class TestAgentGraphStructure:
    def test_graph_built(self, graph):
        assert graph is not None

    def test_nodes_exist(self, graph):
        nodes = list(graph.get_graph().nodes.keys())
        # classify_intent 保留为占位 passthrough 节点（意图分析已移除）
        for node in {"classify_intent", "vision_analyze", "rewrite_query",
                     "react_generate", "verify", "output", "retry"}:
            assert node in nodes, f"节点 {node} 缺失"
        # 意图分析已移除：chitchat 独立节点不再存在
        assert "chitchat" not in nodes, "chitchat 节点应已移除"
        # ReAct 模式已移除独立的 retrieve + generate 节点
        assert "retrieve" not in nodes, "retrieve 节点应在 ReAct 模式中被移除"
        assert "generate" not in nodes, "generate 节点应在 ReAct 模式中被移除"

    def test_edges_exist(self, graph):
        edges = graph.get_graph().edges
        assert len(edges) > 0


class TestBuildInputText:
    """重试时输入拼装：应带回上一轮工具结果，避免从零重探"""

    def test_first_pass_no_prev_context(self):
        from agent.graph import _build_input_text
        state = {"question": "找第一个文件", "context": "一些结果", "retry_count": 0}
        text = _build_input_text(state)
        assert "【上一次已获取的信息】" not in text
        assert "找第一个文件" in text

    def test_retry_injects_prev_context(self):
        from agent.graph import _build_input_text
        state = {
            "question": "找第一个文件",
            "context": "目录列表: a.txt, b.txt",
            "retry_count": 1,
            "verification": {"suggestion": "答案不够明确"},
        }
        text = _build_input_text(state)
        assert "【上一次已获取的信息】" in text
        assert "目录列表: a.txt, b.txt" in text     # 复用上一轮工具结果
        assert "【改进要求】答案不够明确" in text

    def test_retry_skips_empty_context(self):
        from agent.graph import _build_input_text
        state = {"question": "q", "context": "无相关资料", "retry_count": 1}
        text = _build_input_text(state)
        assert "【上一次已获取的信息】" not in text

