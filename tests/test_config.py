"""测试配置模块"""
import importlib
import os

import config.settings


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
        monkeypatch.setenv("APP_PORT", "9090")
        importlib.reload(config.settings)
        from config import settings as cfg
        assert cfg.PORT == 9090
        monkeypatch.delenv("APP_PORT")
        importlib.reload(config.settings)

    def test_otlp_endpoint_defaults_to_collector_paths(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
        importlib.reload(config.settings)
        from config import settings as cfg
        assert cfg.otel_traces_endpoint == "http://localhost:4318/v1/traces"
        assert cfg.otel_metrics_endpoint == "http://localhost:4318/v1/metrics"

    def test_otlp_base_endpoint_expands_per_signal_paths(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
        importlib.reload(config.settings)
        from config import settings as cfg
        assert cfg.otel_traces_endpoint == "http://collector:4318/v1/traces"
        assert cfg.otel_metrics_endpoint == "http://collector:4318/v1/metrics"
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        importlib.reload(config.settings)

    def test_signal_specific_endpoint_overrides_base_endpoint(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:9999/custom/traces")
        importlib.reload(config.settings)
        from config import settings as cfg
        assert cfg.otel_traces_endpoint == "http://collector:9999/custom/traces"
        assert cfg.otel_metrics_endpoint == "http://collector:4318/v1/metrics"
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        importlib.reload(config.settings)
