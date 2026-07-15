"""Token 预算估算与 ReAct 上下文压缩

用途：
- 在调用 LLM 前估算消息列表的 token 占用，判断是否逼近模型上下文窗口。
- 逼近阈值时，把较早的工具调用往返摘要成一段文字，只保留最近若干轮工具交互，
  避免单轮 ReAct 循环里累积的工具返回撑爆上下文。

估算采用字符启发式（中文约 1.5 字/token，其他约 4 字符/token），不依赖具体分词器，
足以支撑"何时压缩"的阈值判断。
"""
import json
import logging
import math

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import settings as config

logger = logging.getLogger(__name__)

# 摘要输入的字符硬上限，防止把超长历史整体塞给摘要模型
_MAX_SUMMARY_INPUT_CHARS = 60000

_SUMMARY_SYSTEM = "你是一个对话摘要助手，只输出摘要本身，不输出任何元描述。"
_SUMMARY_PROMPT = (
    "请把下面这段 Agent 执行过程压缩成简明摘要，保留：\n"
    "1. 用户的关键诉求与目标\n"
    "2. 已经调用了哪些工具、返回的核心结果\n"
    "3. 已经达成的结论\n"
    "4. 仍未解决或待办的问题\n\n"
    "不要逐条复述原文，不要罗列所有工具调用，不要保留无关内容。"
    "输出 1-3 段中文，不要用列表，不要加任何前缀或元描述。\n\n"
    "=== 待压缩的执行过程 ===\n{body}\n=== 待压缩的执行过程（结束）==="
)


def estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数：中文约 1.5 字/token，其余约 4 字符/token。"""
    if not text:
        return 0
    chinese = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - chinese
    return math.ceil(chinese / 1.5 + other / 4.0)


def _flatten_content(content) -> str:
    """把消息 content 归一化为纯文本（兼容多模态 content block 列表）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content) if content else ""


def _content_tokens(content) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                t = part.get("type")
                if t == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif t in ("image_url", "image"):
                    total += 1024  # 图片按粗略固定值计入
            elif isinstance(part, str):
                total += estimate_tokens(part)
        return total
    return 0


def _tool_calls_of(msg):
    tc = getattr(msg, "tool_calls", None)
    return tc or []


def estimate_message_tokens(messages) -> int:
    """估算 LangChain 消息列表的 token 总数。

    计入每条消息的正文、工具调用的名称与参数，以及每条约 4 token 的固定开销
    （role、分隔符等）。
    """
    total = 0
    for m in messages:
        total += _content_tokens(getattr(m, "content", ""))
        for tc in _tool_calls_of(m):
            if isinstance(tc, dict):
                name = tc.get("name", "")
                args = tc.get("args", {})
            else:
                name = getattr(tc, "name", "")
                args = getattr(tc, "args", {})
            total += estimate_tokens(name)
            try:
                total += estimate_tokens(json.dumps(args, ensure_ascii=False))
            except (TypeError, ValueError):
                total += estimate_tokens(str(args))
        total += 4
    return total


def compression_trigger_tokens(context_window: int = None, ratio: float = None) -> int:
    """ReAct 上下文压缩的绝对触发阈值（token）。"""
    window = context_window or config.model_context_window
    r = ratio if ratio is not None else config.compression_trigger_ratio
    return int(window * r)


def _render_old_messages(old_msgs) -> str:
    """把待压缩的早期消息渲染成给摘要模型的纯文本。"""
    parts = []
    total_chars = 0
    for m in old_msgs:
        role = getattr(m, "type", m.__class__.__name__)
        line = f"{str(role).upper()}: {_flatten_content(getattr(m, 'content', ''))}"
        for tc in _tool_calls_of(m):
            if isinstance(tc, dict):
                name, args = tc.get("name", ""), tc.get("args", {})
            else:
                name, args = getattr(tc, "name", ""), getattr(tc, "args", {})
            line += f"\n  工具调用 {name}: {args}"
        parts.append(line)
        total_chars += len(line)
        if total_chars > _MAX_SUMMARY_INPUT_CHARS:
            parts.append("...(超长内容已截断)")
            break
    return "\n\n".join(parts)


async def compact_react_messages(messages, llm, trigger_tokens: int = None,
                                 retain_recent_rounds: int = None):
    """按需压缩 ReAct 循环的消息列表。

    当估算 token 超过阈值时，把较早的工具调用往返摘要成一段文字，只保留最近
    ``retain_recent_rounds`` 轮工具交互，其余折叠进摘要。分割点始终落在"工具调用轮"
    边界（带 tool_calls 的 AIMessage 处），因此不会切断 tool_call / tool_result 配对。

    返回压缩后的新列表；未触发压缩时原样返回入参。
    """
    if not messages:
        return messages

    trigger = trigger_tokens if trigger_tokens is not None else compression_trigger_tokens()
    retain = (retain_recent_rounds if retain_recent_rounds is not None
              else config.react_retain_recent_rounds)
    retain = max(1, retain)

    current = estimate_message_tokens(messages)
    if current < trigger:
        return messages

    # 定位首个 HumanMessage：它之前是 system（可选），之后才是工具调用轮
    human_idx = None
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            human_idx = i
            break
    if human_idx is None:
        return messages  # 结构异常，保守不动

    # 工具调用轮起点：human 之后带 tool_calls 的 AIMessage
    round_starts = [
        i for i in range(human_idx + 1, len(messages))
        if isinstance(messages[i], AIMessage) and _tool_calls_of(messages[i])
    ]
    if len(round_starts) <= retain:
        return messages  # 工具轮数不足，无从压缩

    split_idx = round_starts[len(round_starts) - retain]
    old_msgs = messages[human_idx + 1:split_idx]
    if not old_msgs:
        return messages

    body = _render_old_messages(old_msgs)
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=_SUMMARY_PROMPT.format(body=body)),
        ])
        summary = (getattr(resp, "content", "") or "").strip()
    except Exception as e:
        logger.warning("ReAct 上下文摘要失败，跳过本次压缩: %s", e)
        return messages
    if not summary:
        return messages

    # 重建：把摘要折入原 Human 消息 → 保证只有一条 human 后接工具轮，
    # tool_call/tool_result 配对完整，且无连续同角色消息。
    head = list(messages[:human_idx])  # system（如有）
    orig_content = _flatten_content(messages[human_idx].content)
    merged_human = HumanMessage(content=(
        f"{orig_content}\n\n[以下为已压缩的早期工具调用与中间结果摘要]\n{summary}"
    ))
    rebuilt = head + [merged_human] + list(messages[split_idx:])

    after = estimate_message_tokens(rebuilt)
    logger.info("ReAct 上下文压缩: tokens %d -> %d, 消息 %d -> %d 条",
                current, after, len(messages), len(rebuilt))
    return rebuilt
