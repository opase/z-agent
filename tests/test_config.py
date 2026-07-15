"""测试配置模块"""
import os


class TestConfig:
    def test_defaults(self, config):
        assert config.HOST == "127.0.0.1"
        assert config.PORT == 8080
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.collection_name == "rag"

    def test_model_names(self, config):
        assert config.embedding_model == "text-embedding-v4"
        assert config.chat_model == "glm-5.1"
        assert config.rerank_model == "gte-rerank-v2"
        assert config.classifier_model == "qwen-turbo"

    def test_retrieval_params(self, config):
        assert config.bm25_top_k == 10
        assert config.vector_top_k == 10
        assert config.hybrid_top_k == 10
        assert config.rerank_top_k == 6

    def test_memory_params(self, config):
        assert config.memory_window_size == 10
        assert config.session_timeout_hours == 24

    def test_env_override(self, monkeypatch):
        import importlib
        import config.settings
        monkeypatch.setenv("APP_PORT", "9090")
        importlib.reload(config.settings)
        from config import settings as cfg
        assert cfg.PORT == 9090
        # restore
        monkeypatch.delenv("APP_PORT")
        importlib.reload(config.settings)
