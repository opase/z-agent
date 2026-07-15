"""会话管理服务"""
import uuid
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from memory.conversation import ConversationMemory, Message
from core.session_store import MemoryStore, SessionStore
from config import settings as config
from core import metrics

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    user_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_active: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class SessionManager:
    def __init__(self, llm=None, store: SessionStore = None, light_llm=None):
        self.sessions: dict[str, Session] = {}
        self.timeout = timedelta(hours=config.session_timeout_hours)
        self._llm = llm
        self._light_llm = light_llm  # Phase 3: 压缩用轻量模型
        self.store = store or MemoryStore()

    def _make_on_message(self, session_id: str):
        """创建持久化回调——闭包捕获 session_id"""
        store = self.store

        def on_message(role: str, content: str, image_count: int = 0):
            try:
                store.save_message(session_id, role, content, image_count)
            except Exception as e:
                logger.error("持久化消息失败: %s", e)

        def on_summary(summary: str):
            try:
                store.save_summary(session_id, summary)
            except Exception as e:
                logger.error("持久化摘要失败: %s", e)

        return on_message, on_summary

    def _create_session(self, session_id: str, user_id: str) -> Session:
        """创建新会话并持久化"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        on_msg, on_sum = self._make_on_message(session_id)
        s = Session(
            session_id=session_id, user_id=user_id,
            memory=ConversationMemory(
                light_llm=self._light_llm,
                on_message=on_msg,
                on_summary=on_sum,
                session_id=session_id,
                user_id=user_id,
            ),
            created_at=now, last_active=now,
        )
        self.sessions[session_id] = s
        try:
            self.store.touch_session(session_id, user_id)
        except Exception as e:
            logger.error("持久化会话失败: %s", e)
        metrics.active_sessions.inc()
        return s

    def _restore_session(self, session_id: str, user_id: str) -> Session | None:
        """从持久化存储恢复会话"""
        messages = self.store.get_messages(session_id)
        if not messages:
            return None

        on_msg, on_sum = self._make_on_message(session_id)
        s = Session(
            session_id=session_id, user_id=user_id,
            memory=ConversationMemory(
                light_llm=self._light_llm,
                on_message=on_msg,
                on_summary=on_sum,
                session_id=session_id,
                user_id=user_id,
            ),
        )
        # 恢复历史消息
        for m in messages:
            s.memory.messages.append(Message(
                role=m["role"], content=m["content"],
                image_count=m.get("image_count", 0),
                timestamp=m.get("timestamp", ""),
            ))
        # 恢复摘要
        if hasattr(self.store, "get_summary"):
            summary = self.store.get_summary(session_id)
            if summary:
                s.memory.summary = summary

        s.last_active = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sessions[session_id] = s
        logger.info("从存储恢复会话 %s: %d 条消息", session_id, len(messages))
        return s

    def get(self, session_id: str, user_id: str = "default") -> Session | None:
        """获取已有会话（内存 → 持久化存储），不存在返回 None"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        restored = self._restore_session(session_id, user_id)
        if restored:
            self.sessions[session_id] = restored
            return restored
        # 兜底：0 轮会话在 messages 表无记录，但 sessions 表有记录
        if self.store.session_exists(session_id):
            return self._create_session(session_id, user_id)
        return None

    def delete(self, session_id: str):
        """直接删除会话（不论是否在内存中）"""
        self.sessions.pop(session_id, None)
        try:
            self.store.delete_session(session_id)
        except Exception as e:
            logger.error("删除持久化会话失败: %s", e)

    def get_or_create(self, session_id: str = None, user_id: str = "default") -> Session:
        # 先检查内存
        if session_id and session_id in self.sessions:
            s = self.sessions[session_id]
            last = datetime.strptime(s.last_active, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - last > self.timeout:
                self.remove(session_id)
            else:
                s.last_active = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return s

        # 尝试从持久化存储恢复
        if session_id:
            restored = self._restore_session(session_id, user_id)
            if restored:
                return restored

        # 新建
        sid = session_id or str(uuid.uuid4())[:8]
        return self._create_session(sid, user_id)

    def remove(self, session_id: str):
        """同步移除会话（内部使用：超时清理等）"""
        if session_id in self.sessions:
            self.sessions.pop(session_id, None)
            metrics.active_sessions.dec()
        try:
            self.store.delete_session(session_id)
        except Exception as e:
            logger.error("删除持久化会话失败: %s", e)

    async def safe_remove(self, session_id: str):
        """异步移除会话（API 使用：等待压缩任务完成后删除）"""
        existed = session_id in self.sessions
        session = self.sessions.pop(session_id, None)
        if existed:
            metrics.active_sessions.dec()
        if session and hasattr(session.memory, 'await_compression'):
            await session.memory.await_compression()
        try:
            self.store.delete_session(session_id)
        except Exception as e:
            logger.error("删除持久化会话失败: %s", e)

    def list_all(self) -> list[dict]:
        # 优先用持久化存储的数据（跨重启一致）
        try:
            stored = self.store.list_sessions()
            if stored:
                return stored
        except Exception as e:
            logger.error("获取持久化会话列表失败: %s", e)
        # 兜底：内存数据
        return [{"session_id": s.session_id, "user_id": s.user_id,
                 "turn_count": s.memory.turn_count, "created_at": s.created_at}
                for s in self.sessions.values()]
