"""测试会话管理"""
from core.session_service import SessionManager


class TestSessionManager:
    def test_create_session(self):
        sm = SessionManager()
        session = sm.get_or_create(user_id="user1")
        assert session.user_id == "user1"
        assert session.session_id is not None
        assert len(session.session_id) == 8

    def test_reuse_session(self):
        sm = SessionManager()
        s1 = sm.get_or_create(session_id="abc12345", user_id="user1")
        s2 = sm.get_or_create(session_id="abc12345", user_id="user1")
        assert s1 is s2

    def test_remove_session(self):
        sm = SessionManager()
        sm.get_or_create(session_id="xyz", user_id="u1")
        sm.remove("xyz")
        assert "xyz" not in sm.sessions

    def test_list_all(self):
        sm = SessionManager()
        sm.get_or_create(session_id="s1", user_id="u1")
        sm.get_or_create(session_id="s2", user_id="u2")
        all_sessions = sm.list_all()
        assert len(all_sessions) == 2
        ids = {s["session_id"] for s in all_sessions}
        assert ids == {"s1", "s2"}
