"""Skill 索引格式化 — 注入 system prompt

预算约束：
- 单条 description ≤ 500 codepoint
- 启用 skill 数 ≤ 20
- 总段大小 ≤ 4096 字符
"""
import logging

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_LENGTH = 500
MAX_ENABLED_SKILLS = 20
MAX_INDEX_CHARS = 4096


def format_skill_index(enabled_skills: list) -> str:
    """把启用 skill 渲染成 system prompt 中的 ## 可用 Skills 段落"""
    if not enabled_skills:
        return ""

    effective = enabled_skills
    if len(effective) > MAX_ENABLED_SKILLS:
        effective = sorted(effective, key=lambda s: s.name)[:MAX_ENABLED_SKILLS]
        logger.warning("skill 数 %d 超过上限 %d，仅前 %d 进入索引",
                       len(enabled_skills), MAX_ENABLED_SKILLS, MAX_ENABLED_SKILLS)

    lines = ["## 可用 Skills（按需调用 load_skill 加载完整指引）", ""]
    for skill in sorted(effective, key=lambda s: s.name):
        desc = _truncate(skill.description, MAX_DESCRIPTION_LENGTH)
        tags_str = f" [{', '.join(skill.tags)}]" if skill.tags else ""
        lines.append(f"- **{skill.name}**{tags_str}：{desc}")

    lines.append("")
    lines.append(
        "判断准则：当任务描述匹配某个 skill 的触发场景时，调用 load_skill(name) 加载完整指引；"
        "已加载的 skill 会在下一轮以 \"## 已加载 Skill\" 段落出现在你的 user message 中。"
        "不要重复加载同一 skill；同一会话内一次足够。"
    )

    result = "\n".join(lines)
    if len(result) > MAX_INDEX_CHARS:
        result = result[:MAX_INDEX_CHARS] + "\n...(skill 索引段被截断)"
        logger.warning("skill 索引段超过 %d 字符，已截断", MAX_INDEX_CHARS)

    return result


def _truncate(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "..."
