"""Skill MCP 工具 — load_skill / list_skills"""
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 全局引用，由 RagService 注入
_registry = None
_buffer_by_thread: dict[str, "SkillContextBuffer"] = {}


def set_skill_registry(registry):
    global _registry
    _registry = registry


def get_buffer(thread_id: str) -> "SkillContextBuffer":
    from .buffer import SkillContextBuffer

    if thread_id not in _buffer_by_thread:
        _buffer_by_thread[thread_id] = SkillContextBuffer()
    return _buffer_by_thread[thread_id]


def clear_buffer(thread_id: str):
    buf = _buffer_by_thread.pop(thread_id, None)
    if buf:
        buf.clear()


# ══════════════════════════════════════════════════════════
# LangChain Tools
# ══════════════════════════════════════════════════════════


@tool
def load_skill(name: str) -> str:
    """加载指定的 skill 到上下文。当你需要执行特定领域的任务时，先调用此工具加载对应 skill。

    可用 skill 列表可通过 list_skills 工具查询。
    加载后 skill 的指导内容会自动注入到后续对话中。
    """
    if _registry is None:
        return "[错误] Skill 系统未初始化"

    skill = _registry.find(name)
    if skill is None:
        available = [s.name for s in _registry.enabled_skills()]
        return f"[未找到] skill '{name}' 不存在或已禁用。可用: {', '.join(available) if available else '(无)'}"

    # 不在此处 push buffer（没有 thread_id），由 _react_generate 在工具结果中处理
    return f"[已加载] {skill.name}: {skill.description}\n\n{skill.body}"


@tool
def list_skills() -> str:
    """列出所有可用的 skill 名称和描述"""
    if _registry is None:
        return "[错误] Skill 系统未初始化"

    skills = _registry.enabled_skills()
    if not skills:
        return "当前没有可用的 skill。"

    lines = []
    for s in skills:
        tags_str = f" [{', '.join(s.tags)}]" if s.tags else ""
        src = f" ({s.display_source})"
        lines.append(f"- **{s.name}**{tags_str}{src}: {s.description}")
    return "\n".join(lines)
