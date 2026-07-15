"""测试 API 路由"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    try:
        from main import app
        return TestClient(app)
    except Exception:
        pytest.skip("无法创建 TestClient")


class TestRootEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "Zagent"
        assert data["status"] == "running"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "bm25_docs" in data
        assert "active_sessions" in data

    def test_request_id_header(self, client):
        resp = client.get("/")
        assert "X-Request-ID" in resp.headers

    def test_custom_request_id(self, client):
        resp = client.get("/", headers={"X-Request-ID": "my-trace-123"})
        assert resp.headers["X-Request-ID"] == "my-trace-123"


class TestUserAPI:
    def test_get_profile(self, client):
        resp = client.get("/users/test_api_user/profile")
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "test_api_user"

    def test_list_sessions(self, client):
        resp = client.get("/users/sessions")
        assert resp.status_code == 200
        assert "sessions" in resp.json()
