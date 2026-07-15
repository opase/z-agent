"""Skill 上下文缓冲区 — LLM 调用 load_skill 后注入上下文

生命周期：LLM 调 load_skill → push → 下一轮 drain() → 拼到 user message 前

关键约束：
- drain() 一次性消费（防跨轮重复注入）
- 同一会话最多保留 3 个 skill body（LRU 淘汰）
- 同一 skill 重复 push 替换旧 body
"""
from collections import OrderedDict


class SkillContextBuffer:
    """单线程（单 Agent 实例）的 skill 注入缓冲区"""

    MAX_SKILLS = 3

    def __init__(self):
        self._entries: OrderedDict[str, str] = OrderedDict()

    def push(self, skill_name: str, body: str):
        if not skill_name or not body:
            return
        # 同名替换 → 移到末尾（LRU）
        self._entries.pop(skill_name, None)
        self._entries[skill_name] = body
        # 超出上限 → 淘汰最旧
        while len(self._entries) > self.MAX_SKILLS:
            self._entries.popitem(last=False)

    def drain(self) -> str:
        """取出全部并清空，返回拼好的 markdown 段"""
        if not self._entries:
            return ""
        items = list(self._entries.items())
        self._entries.clear()

        parts = []
        for name, body in items:
            parts.append(f"## 已加载 Skill：{name}\n{body.strip()}\n")
        parts.append("---")
        return "\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    @property
    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()
