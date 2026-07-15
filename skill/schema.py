"""Skill 数据结构"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Source(Enum):
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"


@dataclass
class Skill:
    """一个 Skill 是沉淀决策与经验的复用单元。

    由 SKILL.md 文件解析得到：frontmatter 决定元数据，
    body 在 LLM 调用 load_skill 时通过 SkillContextBuffer 注入上下文。
    """
    name: str
    description: str = ""
    version: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    source: Source = Source.PROJECT
    body: str = ""
    skill_md_path: Path | None = None
    references_dir: Path | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Skill name 不能为空")

    @property
    def display_source(self) -> str:
        return self.source.value
