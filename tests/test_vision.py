"""测试多模态视觉模块"""
import pytest


class TestVisionAnalyzer:
    def test_build_messages(self):
        from agent.vision import VisionAnalyzer
        va = VisionAnalyzer()
        msgs = va._build_messages(["img1_base64"], "这是什么手机？")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        user_content = msgs[1]["content"]
        assert any("image" in (item if isinstance(item, dict) else "") or item.get("image") for item in user_content if isinstance(item, dict))

    def test_parse_valid_json(self):
        from agent.vision import VisionAnalyzer
        result = VisionAnalyzer._parse_response(
            '{"description":"一部黑色手机","detected_products":["小米15"],"detected_issues":[],"scene_type":"product_photo"}'
        )
        assert result["description"] == "一部黑色手机"
        assert result["detected_products"] == ["小米15"]
        assert result["scene_type"] == "product_photo"

    def test_parse_json_with_extra_text(self):
        from agent.vision import VisionAnalyzer
        result = VisionAnalyzer._parse_response(
            '这是分析结果：{"description":"报错截图","detected_products":[],"detected_issues":["系统崩溃"],"scene_type":"error_screenshot"}'
        )
        assert result["description"] == "报错截图"
        assert result["detected_issues"] == ["系统崩溃"]
        assert result["scene_type"] == "error_screenshot"

    def test_parse_invalid_returns_default(self):
        from agent.vision import VisionAnalyzer
        result = VisionAnalyzer._parse_response("not json at all")
        assert result["description"] == ""
        assert result["detected_products"] == []
        assert result["scene_type"] == "other"

    def test_empty_result(self):
        from agent.vision import VisionAnalyzer
        result = VisionAnalyzer._empty()
        assert result["description"] == ""
        assert result["detected_products"] == []
        assert result["detected_issues"] == []
        assert result["scene_type"] == "other"

    def test_model_config(self):
        from agent.vision import VisionAnalyzer
        from config import settings as config
        va = VisionAnalyzer()
        assert va.model == config.vision_model
        assert va.model == "qwen3-vl"


class TestVisionIntegration:
    def test_vision_node_in_graph(self):
        from core.rag_service import RagService
        rag = RagService()
        nodes = list(rag.agent_graph.get_graph().nodes.keys())
        assert "vision_analyze" in nodes

    def test_agent_state_has_image_fields(self):
        from agent.graph import AgentState
        assert "images" in AgentState.__annotations__
        assert "image_desc" in AgentState.__annotations__
        assert "detected_products" in AgentState.__annotations__

    def test_message_has_image_count(self):
        from memory.conversation import Message
        msg = Message(role="user", content="这是什么？", image_count=2)
        assert msg.image_count == 2

    def test_context_string_includes_image_marker(self):
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_message("user", "这是什么？", image_count=2)
        ctx = mem.get_context_string()
        assert "[附图2张]" in ctx
