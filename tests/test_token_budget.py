"""测试 token 预算估算与 ReAct 上下文压缩"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
)

from memory.token_budget import (
    estimate_tokens, estimate_message_tokens,
    compression_trigger_tokens, compact_react_messages,
)


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_chinese_ratio(self):
        # 3 个中文 → ceil(3/1.5) = 2
        assert estimate_tokens("你好啊") == 2

    def test_ascii_ratio(self):
        # 8 个 ascii → ceil(8/4) = 2
        assert estimate_tokens("abcdefgh") == 2

    def test_mixed(self):
        # 2 中文 + 4 ascii → ceil(2/1.5 + 4/4) = ceil(2.33) = 3
        assert estimate_tokens("你好abcd") == 3

    def test_longer_text_more_tokens(self):
        assert estimate_tokens("你好" * 100) > estimate_tokens("你好")


class TestEstimateMessageTokens:
    def test_counts_content_and_overhead(self):
        msgs = [HumanMessage(content="你好啊")]  # 2 token + 4 开销
        assert estimate_message_tokens(msgs) == 6

    def test_counts_tool_call_args(self):
        ai = AIMessage(content="", tool_calls=[{
            "name": "read_file", "args": {"path": "abcd"}, "id": "t1",
        }])
        # 有工具调用参数 → token 大于纯空内容的固定开销
        assert estimate_message_tokens([ai]) > 4

    def test_multimodal_content_list(self):
        msg = HumanMessage(content=[
            {"type": "text", "text": "你好啊"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ])
        # 文本 2 + 图片 1024 + 开销 4
        assert estimate_message_tokens([msg]) >= 1024


class TestCompressionTrigger:
    def test_trigger_from_window(self):
        assert compression_trigger_tokens(1000, 0.9) == 900


def _mock_summary_llm(text="早期过程摘要"):
    m = MagicMock()
    m.ainvoke = AsyncMock(return_value=MagicMock(content=text))
    return m


def _build_react_messages(num_rounds: int, tool_chars: int = 10):
    """构造 [System, Human, (AI(tool_calls)+Tool) × N] 结构"""
    msgs = [SystemMessage(content="sys"), HumanMessage(content="原始问题")]
    for i in range(num_rounds):
        msgs.append(AIMessage(content="", tool_calls=[{
            "name": "search", "args": {"q": f"query_{i}"}, "id": f"t{i}",
        }]))
        msgs.append(ToolMessage(content="结" * tool_chars, tool_call_id=f"t{i}"))
    return msgs


class TestCompactReactMessages:
    def test_below_trigger_no_change(self):
        msgs = _build_react_messages(2)
        llm = _mock_summary_llm()

        async def _run():
            out = await compact_react_messages(msgs, llm, trigger_tokens=10_000_000)
            return out

        out = asyncio.run(_run())
        assert out is msgs                 # 未触发 → 原样返回
        assert llm.ainvoke.await_count == 0  # 未调用摘要

    def test_compress_keeps_recent_rounds(self):
        msgs = _build_react_messages(10, tool_chars=200)
        llm = _mock_summary_llm()

        async def _run():
            return await compact_react_messages(
                msgs, llm, trigger_tokens=1, retain_recent_rounds=3,
            )

        out = asyncio.run(_run())
        # 调用了摘要
        assert llm.ainvoke.await_count == 1
        # 结构：[System, MergedHuman, (AI+Tool) × 3]
        assert isinstance(out[0], SystemMessage)
        assert isinstance(out[1], HumanMessage)
        ai_rounds = [m for m in out if isinstance(m, AIMessage)]
        assert len(ai_rounds) == 3
        # 摘要折入 human，原问题仍在
        assert "原始问题" in out[1].content
        assert "早期过程摘要" in out[1].content

    def test_tool_pairs_not_split(self):
        """压缩后每个带 tool_calls 的 AIMessage 后面都紧跟对应 ToolMessage"""
        msgs = _build_react_messages(8, tool_chars=200)
        llm = _mock_summary_llm()

        async def _run():
            return await compact_react_messages(
                msgs, llm, trigger_tokens=1, retain_recent_rounds=2,
            )

        out = asyncio.run(_run())
        for i, m in enumerate(out):
            if isinstance(m, AIMessage) and m.tool_calls:
                ids = {tc["id"] for tc in m.tool_calls}
                # 紧随其后的 ToolMessage 覆盖该 AIMessage 的所有 tool_call_id
                following_ids = set()
                j = i + 1
                while j < len(out) and isinstance(out[j], ToolMessage):
                    following_ids.add(out[j].tool_call_id)
                    j += 1
                assert ids <= following_ids, f"第 {i} 条 AIMessage 的 tool 配对被切断"

    def test_not_enough_rounds_skip(self):
        msgs = _build_react_messages(3)
        llm = _mock_summary_llm()

        async def _run():
            return await compact_react_messages(
                msgs, llm, trigger_tokens=1, retain_recent_rounds=3,
            )

        out = asyncio.run(_run())
        assert out is msgs                 # 轮数不足 retain → 不压缩
        assert llm.ainvoke.await_count == 0

    def test_summary_failure_returns_original(self):
        msgs = _build_react_messages(10, tool_chars=200)
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        async def _run():
            return await compact_react_messages(msgs, llm, trigger_tokens=1)

        out = asyncio.run(_run())
        assert out is msgs                 # 摘要失败 → 原样返回，不破坏消息
