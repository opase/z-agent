"""OpenTelemetry tracing/metrics → Arize Phoenix (OTLP/HTTP)。

取代 core/metrics.py。Span/Metric 经 OTLP 发往 Phoenix；Phoenix 接收后自动聚合，
并通过自身 /metrics 端点暴露 Prometheus 格式供抓取 —— Python 侧无需为 Prometheus 写任何代码。

架构（Phoenix 为主、Prometheus 为辅）：
- 零侵入自动埋点：LangChainInstrumentor 拦截全部 LLM 调用点（自动采集 token）；
  FastAPIInstrumentor（在 main.py 注册）自动生成 HTTP server span。
- 手动骨架 Span：仅在 Agent 入口（z_agent.chat）与工具咽喉点（tool.invoke）写业务语义
  （task.status / error.type / user.query）。
- OTel Metrics API：仅保留 2 个瞬时 Gauge（active_sessions / mcp_connection_status），
  其余计数器由 Phoenix 从 span 派生。
"""
import re
import logging
import functools
import inspect
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from config import settings as config

logger = logging.getLogger(__name__)

# 幂等标记：setup_tracing 重复调用安全
_PROVIDER_OK = False


def _patch_tracer_interrupt_hooks() -> None:
    """兜底 openinference-instrumentation-langchain 0.1.67 对 LangGraph 1.x 的回调缺口。

    该版本的 OpenInferenceTracer 未实现 on_interrupt / on_resume，而 LangChain 1.x 的
    回调管理器在 HITL interrupt()/resume() 时会直接调用 handler.on_interrupt(...)（不走
    安全 getattr），缺方法即抛 AttributeError。报错会打断 tracer 的 span 栈记账，导致
    resume 后的 span、被打断的父 span 的 input/output 写不进去（Phoenix 显示空）。

    上游（>=0.1.68 若实现）修复后可移除本兜底。
    """
    try:
        from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
    except Exception:
        return
    for method in ("on_interrupt", "on_resume"):
        if not hasattr(OpenInferenceTracer, method):
            setattr(OpenInferenceTracer, method, lambda self, *args, **kwargs: None)


def _patch_tongyi_usage_metadata() -> None:
    """补 ChatTongyi 的 AIMessage.usage_metadata，使 token 能被 instrumentor 采到。

    根因：langchain_community 的 ChatTongyi 把 token 放在 response_metadata.token_usage，
    未规范化到 AIMessage.usage_metadata；而 openinference-instrumentation-langchain
    只读 usage_metadata → llm.token_count.* 写不进 span（token 数据其实已随 output.value
    序列化进去，但属性栏空）。把 token_usage 复制到 usage_metadata（LangChain 标准字段）
    即可，instrumentor 一定读得到。幂等。
    """
    try:
        from langchain_community.chat_models import ChatTongyi
    except Exception:
        return

    def _fill(result):
        try:
            # generations 可能是 List[ChatGeneration]（langchain 1.x 实测为单层）
            # 或 List[List[ChatGeneration]]，兼容两种
            gens = getattr(result, "generations", []) or []
            flat = []
            for item in gens:
                if hasattr(item, "message"):
                    flat.append(item)
                elif hasattr(item, "__iter__"):
                    flat.extend(x for x in item if hasattr(x, "message"))
            for gen in flat:
                msg = getattr(gen, "message", None)
                gi = getattr(gen, "generation_info", None) or {}
                tu = gi.get("token_usage") if isinstance(gi, dict) else None
                if not tu and msg is not None:
                    tu = (getattr(msg, "response_metadata", None) or {}).get("token_usage")
                if not tu:
                    continue
                um = {
                    "input_tokens": int(tu.get("input_tokens") or tu.get("prompt_tokens") or 0),
                    "output_tokens": int(tu.get("output_tokens") or tu.get("completion_tokens") or 0),
                    "total_tokens": int(tu.get("total_tokens") or 0),
                }
                # 同时填 message.usage_metadata（instrumentor 路径2）和
                # generation_info["usage_metadata"]（路径3），_token_counts 才能命中
                if msg is not None and not getattr(msg, "usage_metadata", None):
                    msg.usage_metadata = um
                if isinstance(gi, dict) and "usage_metadata" not in gi:
                    gi["usage_metadata"] = um
        except Exception:
            pass
        return result

    def _patch(name, is_async):
        orig = getattr(ChatTongyi, name, None)
        if orig is None or getattr(orig, "_z_agent_usage_patched", False):
            return

        if is_async:
            @functools.wraps(orig)
            async def _aw(self, *a, **kw):
                r = await orig(self, *a, **kw)
                return _fill(r)
            _aw._z_agent_usage_patched = True
            setattr(ChatTongyi, name, _aw)
        else:
            @functools.wraps(orig)
            def _w(self, *a, **kw):
                r = orig(self, *a, **kw)
                return _fill(r)
            _w._z_agent_usage_patched = True
            setattr(ChatTongyi, name, _w)

    _patch("_generate", is_async=False)
    _patch("_agenerate", is_async=inspect.iscoroutinefunction(getattr(ChatTongyi, "_agenerate", None)))


def setup_tracing() -> None:
    """初始化 TracerProvider / MeterProvider 并挂载 LangChain 自动 instrumentation。

    幂等且绝不抛异常：缺 Phoenix 或 OTel 关闭时导出静默失败，业务不受影响。
    """
    global _PROVIDER_OK
    if _PROVIDER_OK or not config.otel_enabled:
        return
    try:
        resource = Resource.create({
            "service.name": config.otel_service_name,
            "service.version": "2.0.0",
            "deployment.environment": config.otel_environment,
        })

        # ── Traces ──
        tp = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=config.phoenix_traces_endpoint,            # 必须含 /v1/traces 路径
            headers=_auth_headers(),
        )
        tp.add_span_processor(BatchSpanProcessor(exporter))
        if config.otel_debug_console:
            tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        _set_provider_safe(trace.set_tracer_provider, tp)

        # ── Metrics（2 个瞬时 Gauge 经 OTLP 发送，可选）──
        # 旧版 Phoenix 不摄取 OTLP metrics（/v1/metrics 返回 405），默认关闭避免刷屏 ERROR；
        # 关闭时 get_meter() 返回默认空 meter，UpDownCounter/ObservableGauge 退化为 no-op。
        if config.otel_metrics_enabled:
            mreader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=config.phoenix_metrics_endpoint, headers=_auth_headers()),
                export_interval_millis=10000,
            )
            _set_provider_safe(metrics.set_meter_provider,
                               MeterProvider(resource=resource, metric_readers=[mreader]))
            logger.info("OTel metrics 已启用 → %s", config.phoenix_metrics_endpoint)
        else:
            logger.info("OTel metrics 未启用（OTEL_METRICS_ENABLED=false）；traces 照常工作，Gauge 退化为 no-op")

        # ── 自动 instrumentation：LangChain（覆盖全部 LLM 调用点 + token 采集）──
        # FastAPI 自动埋点需访问 app 实例，由 main.py 在 app 创建后调用。
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument()
        _patch_tracer_interrupt_hooks()  # 兜底 HITL interrupt/resume 回调缺口
        _patch_tongyi_usage_metadata()   # 补 ChatTongyi usage_metadata，使 token 可被采集

        _PROVIDER_OK = True
        logger.info("OTel tracing 已启用 → %s", config.phoenix_traces_endpoint)
    except Exception as e:
        # 监控降级，不影响业务
        logger.warning("OTel 初始化失败（监控降级，业务不受影响）: %s", e)
        _PROVIDER_OK = False


def _auth_headers():
    return {"authorization": f"Bearer {config.phoenix_api_key}"} if config.phoenix_api_key else None


def _set_provider_safe(setter, provider) -> None:
    """set_*_provider 重复调用会抛 'Overriding ... is not allowed'，此处吞掉以保证幂等。"""
    try:
        setter(provider)
    except Exception as e:
        if "Overriding" not in str(e):
            raise


# ── OpenInference 语义属性键（与 LangChainInstrumentor 自动写入一致）──
# 手写 span（z_agent.chat / tool.invoke 等）若不设这些键，Phoenix 列表的
# kind/input/output/status 列会显示 '-'。下面常量 + set_io/mark_success 用于补齐。
SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME_TYPE = "output.mime_type"
KIND_CHAIN = "CHAIN"
KIND_AGENT = "AGENT"
KIND_TOOL = "TOOL"

_IO_MAX = 2000  # input/output.value 单段最大字符，避免巨型属性拖慢后端


def _truncate(text, limit: int = _IO_MAX) -> str:
    s = str(text)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


def set_io(span_, *, kind: str = None, input_value=None, output_value=None,
           mime: str = "text/plain") -> None:
    """给手写 span 注入 OpenInference 的 kind/input/output，使 Phoenix 不再显示 '-'。"""
    if kind:
        span_.set_attribute(SPAN_KIND, kind)
    if input_value is not None:
        span_.set_attribute(INPUT_VALUE, _truncate(input_value))
        span_.set_attribute(INPUT_MIME_TYPE, mime)
    if output_value is not None:
        span_.set_attribute(OUTPUT_VALUE, _truncate(output_value))
        span_.set_attribute(OUTPUT_MIME_TYPE, mime)


def mark_success(span_, output_value=None) -> None:
    """成功收尾：设 StatusCode.OK（Phoenix status 列读这个，不是 task.status 属性）+ 可选 output。"""
    span_.set_attribute("task.status", "success")
    span_.set_status(Status(StatusCode.OK))
    if output_value is not None:
        span_.set_attribute(OUTPUT_VALUE, _truncate(output_value))
        span_.set_attribute(OUTPUT_MIME_TYPE, "text/plain")


def end_span(span_, *, ok: bool, output_value=None, note: str = None) -> None:
    """手动收尾一个非 with 块管理的 span（如 chat.ttft）：设状态 + end。

    ok=False 用于"无产出"场景（如审批中断未吐字），状态留 UNSET、记 note，不算成功也不算失败。
    """
    if ok:
        mark_success(span_, output_value=output_value)
    elif note:
        span_.set_attribute("task.status", note)
    span_.end()


def set_ablation(span_) -> None:
    """写消融实验标签到 span（experiment.id / prompt.version / retrieval.scheme / tool_desc.version）。

    Phoenix 按这些属性 group-by 即可对比不同 Prompt / 检索方案 / 工具描述的效果。
    """
    span_.set_attribute("experiment.id", config.experiment_id)
    span_.set_attribute("prompt.version", config.prompt_version)
    span_.set_attribute("retrieval.scheme", config.retrieval_scheme)
    span_.set_attribute("tool_desc.version", config.tool_desc_version)


def shutdown_tracing() -> None:
    """优雅关闭：刷盘 in-flight span / metric。"""
    for getter in (trace.get_tracer_provider, metrics.get_meter_provider):
        prov = getter()
        if hasattr(prov, "shutdown"):
            try:
                prov.shutdown()
            except Exception:
                pass


def get_tracer():
    return trace.get_tracer("z_agent")


def get_meter():
    return metrics.get_meter("z_agent")


def current_span():
    """返回当前激活的 span（无激活时返回无效 span，set_attribute 安全无副作用）。"""
    return trace.get_current_span()


@contextmanager
def span(name: str, **attrs):
    """手动骨架 span 上下文管理器。

    自动在正常退出时设 task.status=success，异常时设 task.status=failure 并 record_exception。
    用法：
        with tracing.span("tool.invoke", **{"tool.name": name}) as s:
            ...
    """
    with get_tracer().start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v)
        try:
            yield s
            s.set_attribute("task.status", "success")
            s.set_status(Status(StatusCode.OK))  # Phoenix status 列读 StatusCode（UNSET 会显示 '-'）
        except Exception as exc:
            s.set_attribute("task.status", "failure")
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            raise


def record_error(span_, error_type: str, exc: BaseException = None) -> None:
    """异常被吞（如 MCP call_tool 返回字符串不 re-raise）时手动归类失败。

    span() 仅在异常 re-raise 时自动标 failure；吞异常路径必须显式调用本函数。
    同时设 StatusCode.ERROR，Phoenix 的 status/error 列才会显示失败。
    """
    span_.set_attribute("task.status", "failure")
    span_.set_attribute("error.type", error_type)
    span_.set_status(Status(StatusCode.ERROR, str(exc) if exc is not None else error_type))
    if exc is not None:
        span_.record_exception(exc)


def sanitize_query(q: str, max_len: int = 200) -> str:
    """脱敏用户问题后再写入 user.query 属性：截断 + 抹掉 API key / 邮箱。"""
    if not q:
        return ""
    redacted = re.sub(
        r"(sk-[A-Za-z0-9]{8,}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+)",
        "***", q,
    )
    return redacted[:max_len]
