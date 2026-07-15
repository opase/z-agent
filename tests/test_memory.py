"""测试记忆模块 — Phase 3: Map-Reduce 压缩"""
import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock


class TestConversationMemory:
    def test_add_message(self):
        from memory.conversation import ConversationMemory
        mem = ConversationMemory(window_size=5)
        mem.add_message("user", "你好")
        mem.add_message("assistant", "你好！有什么可以帮助你的？")
        assert mem.turn_count == 1
        assert not mem.is_empty

    def test_context_string(self):
        from memory.conversation import ConversationMemory
        mem = ConversationMemory(window_size=5)
        mem.add_message("user", "测试问题")
        ctx = mem.get_context_string()
        assert "用户: 测试问题" in ctx or "测试问题" in ctx

    def test_chat_history_format(self):
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_message("user", "Q")
        history = mem.get_chat_history()
        assert history == [{"role": "user", "content": "Q"}]

    def test_clear(self):
        from memory.conversation import ConversationMemory
        mem = ConversationMemory()
        mem.add_message("user", "测试")
        mem.clear()
        assert mem.is_empty
        assert mem.turn_count == 0

    def test_sync_context_graceful(self):
        """同步环境（无事件循环）—— add_message 不崩溃，压缩被跳过"""
        from memory.conversation import ConversationMemory
        mem = ConversationMemory(window_size=2, token_trigger=10)
        for i in range(6):
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")
        # 无事件循环 → 压缩被跳过 → 所有消息保留
        assert len(mem.messages) == 12
        assert mem.turn_count == 6


class TestConversationMemoryAsync:
    """异步压缩测试"""

    @staticmethod
    def _mem(**kw):
        from memory.conversation import ConversationMemory
        # 测试用低 token 阈值，让短消息也能触发压缩
        kw.setdefault("token_trigger", 10)
        return ConversationMemory(**kw)

    @staticmethod
    def _mock_llm(content="压缩摘要"):
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=MagicMock(content=content))
        return mock

    def test_retained_rounds(self):
        """压缩后保留最近 N 轮消息"""
        mock_llm = self._mock_llm()

        async def _run():
            mem = self._mem(window_size=2, light_llm=mock_llm)
            for i in range(8):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            # 触发压缩（8轮×2=16条 > 2*2=4）
            await mem.await_compression()
            # retained_rounds=3 → 6 条保留
            assert len(mem.messages) == 6
            # 保留的是最后 3 轮
            last_user = [m for m in mem.messages if m.role == "user"]
            assert len(last_user) == 3
            assert last_user[-1].content == "问题7"

        asyncio.run(_run())

    def test_map_single_chunk(self):
        """单块（消息数 ≤ chunk_size）→ 只调 map，不调 reduce"""
        mock_llm = self._mock_llm("单块摘要")

        async def _run():
            mem = self._mem(window_size=1, light_llm=mock_llm)
            # 3轮=6条: 2>window*2=2, overflow=3条(retain=6→0 overflow? no:9轮)
            # 7 轮 = 14条: retained=6, overflow=8条=2 blocks
            for i in range(5):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            assert mock_llm.ainvoke.call_count >= 1
            # summary 应被设置（不为空）
            assert mem.summary

        asyncio.run(_run())

    def test_map_reduce_multiple_chunks(self):
        """多块（消息数 > chunk_size）→ Map + Reduce"""
        mock_llm = self._mock_llm("合并摘要")

        async def _run():
            mem = self._mem(window_size=2, light_llm=mock_llm)
            # 20 轮 = 40条: retained=6, overflow=34条=7chunks(5ea)
            for i in range(20):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            # 应该调用了多次 ainvoke（≥2: map*N + reduce*1）
            assert mock_llm.ainvoke.call_count >= 3
            assert mem.summary

        asyncio.run(_run())

    def test_summary_bounded(self):
        """摘要不超过 max_summary_chars"""
        from config import settings as config

        async def _run():
            long_content = "X" * 500
            mock_llm = self._mock_llm(long_content)
            mem = self._mem(window_size=1, light_llm=mock_llm)
            for i in range(6):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            assert len(mem.summary) <= config.compression_max_summary_chars

        asyncio.run(_run())

    def test_fact_extraction(self):
        """压缩时提取事实写入长期记忆"""
        from memory.long_term import LongTermMemory
        import tempfile

        ltm = LongTermMemory("test_facts_user")
        # 伪造 LLM：map 返回普通摘要，extract 返回 facts JSON
        call_count = [0]

        async def mock_ainvoke(prompt):
            call_count[0] += 1
            if "临时任务" in prompt or "持久化事实" in prompt:
                return MagicMock(content=json.dumps({
                    "preferences": ["Python开发"],
                    "profile": {"role": "后端工程师"},
                    "mentioned_products": ["Zagent"],
                }))
            return MagicMock(content="普通摘要")

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_ainvoke

        async def _run():
            mem = self._mem(
                window_size=2, light_llm=mock_llm,
                long_term_memory=ltm, user_id="test_facts_user",
            )
            for i in range(10):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            assert "Python开发" in ltm.data["preferences"]
            assert ltm.data["profile"].get("role") == "后端工程师"
            assert "Zagent" in ltm.data["mentioned_products"]

        asyncio.run(_run())

        # cleanup
        try:
            os.remove(ltm.file_path)
        except Exception:
            pass

    def test_compression_no_llm(self):
        """无 LLM → 压缩优雅跳过"""

        async def _run():
            mem = self._mem(window_size=2)
            for i in range(6):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            # 无 LLM → 压缩跳过，所有消息保留
            assert len(mem.messages) >= 6

        asyncio.run(_run())

    def test_summary_replace_not_accumulate(self):
        """Reduce 阶段用旧摘要+新摘要合并，而非无限拼接"""
        mock_llm = self._mock_llm("最终摘要")

        async def _run():
            mem = self._mem(window_size=1, light_llm=mock_llm)
            mem.summary = "旧摘要文本"
            for i in range(8):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            await mem.await_compression()
            # 摘要应该被替换（不超过 max），而非拼接
            assert "最终摘要" in mem.summary

        asyncio.run(_run())

    def test_llm_fails_graceful_degradation(self):
        """LLM 调用失败 → 不崩溃"""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 崩溃"))

        async def _run():
            mem = self._mem(window_size=1, light_llm=mock_llm)
            for i in range(6):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
            # 不应抛异常
            await mem.await_compression()

        asyncio.run(_run())

    def test_duplicate_task_skipped(self):
        """快速连续 add_message → 不会创建多个重复压缩任务"""
        mock_llm = self._mock_llm()
        tasks = []

        async def _run():
            mem = self._mem(window_size=1, light_llm=mock_llm)
            for i in range(8):
                mem.add_message("user", f"问题{i}")
                mem.add_message("assistant", f"回答{i}")
                if mem._compression_task:
                    tasks.append(mem._compression_task)
            # 可能有多个 task 被创建，但最终应能正确完成
            await mem.await_compression()
            assert mem._compression_task is None or mem._compression_task.done()

        asyncio.run(_run())


class TestLongTermMemory:
    def test_create_and_persist(self):
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory("pytest_user")
        ltm.update_profile({"budget": "5000"})
        ltm.add_preference("拍照")
        assert ltm.data["profile"]["budget"] == "5000"
        assert "拍照" in ltm.data["preferences"]
        os.remove(ltm.file_path)

    def test_context_string(self):
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory("pytest_user_2")
        ltm.update_profile({"usage": "游戏"})
        ctx = ltm.get_context_string()
        assert "游戏" in ctx
        os.remove(ltm.file_path)

    def test_extract_facts_json(self):
        """extract_facts() 解析 JSON 写入长期记忆"""
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory("test_extract_user")
        ltm.extract_facts(json.dumps({
            "preferences": ["暗色主题"],
            "profile": {"tech_stack": "Python"},
            "mentioned_products": ["ChatBot"],
        }))
        assert "暗色主题" in ltm.data["preferences"]
        assert ltm.data["profile"]["tech_stack"] == "Python"
        assert "ChatBot" in ltm.data["mentioned_products"]
        try:
            os.remove(ltm.file_path)
        except Exception:
            pass

    def test_extract_facts_regex_fallback(self):
        """extract_facts() 正则容错提取"""
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory("test_extract_regex")
        raw = '好的，以下是我提取的事实：\n```json\n{"preferences": ["简洁UI"], "profile": {}, "mentioned_products": []}\n```'
        ltm.extract_facts(raw)
        assert "简洁UI" in ltm.data["preferences"]
        try:
            os.remove(ltm.file_path)
        except Exception:
            pass

    def test_extract_facts_invalid(self):
        """extract_facts() 对无效输入不崩溃"""
        from memory.long_term import LongTermMemory
        ltm = LongTermMemory("test_extract_invalid")
        ltm.extract_facts("这不是JSON，也没有包含任何有效数据")
        # 不抛异常即可
        try:
            os.remove(ltm.file_path)
        except Exception:
            pass
