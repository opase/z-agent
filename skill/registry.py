"""Skill 注册表 — 三层目录扫描 + 加载

三层覆盖顺序（后者覆盖前者同名 skill）：
  1. builtin — data/skills/builtin/<name>/SKILL.md
  2. user    — ~/.zagent/skills/<name>/SKILL.md
  3. project — .zagent/skills/<name>/SKILL.md
"""
import logging
from pathlib import Path
from .schema import Skill, Source
from .parser import parse
from .state_store import SkillStateStore

logger = logging.getLogger(__name__)


class SkillRegistry:

    def __init__(
        self,
        builtin_dir: Path | str | None = None,
        user_dir: Path | str | None = None,
        project_dir: Path | str | None = None,
        state_store: SkillStateStore | None = None,
    ):
        from config import settings as config

        self._builtin_dir = Path(builtin_dir) if builtin_dir else Path(config.DATA_DIR) / "skills" / "builtin"
        self._user_dir = Path(user_dir) if user_dir else Path.home() / ".zagent" / "skills"
        self._project_dir = Path(project_dir) if project_dir else Path(".zagent") / "skills"
        self._state_store = state_store or SkillStateStore()
        self._skills: dict[str, Skill] = {}
        self._warnings: list[str] = []

    # ── 加载 ──────────────────────────────────────────────

    def reload(self):
        self._skills.clear()
        self._warnings.clear()
        self._load_dir(self._builtin_dir, Source.BUILTIN)
        self._load_dir(self._user_dir, Source.USER)
        self._load_dir(self._project_dir, Source.PROJECT)
        logger.info("Skill 注册表加载完成: %d 个 skill", len(self._skills))

    def _load_dir(self, dir_path: Path, source: Source):
        if not dir_path.is_dir():
            return
        for entry in sorted(dir_path.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            skill = self._parse_skill(entry, skill_md, source)
            if skill:
                self._skills[skill.name] = skill

    def _parse_skill(self, skill_dir: Path, skill_md: Path, source: Source) -> Skill | None:
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            self._warnings.append(f"读取 SKILL.md 失败 {skill_md}: {e}")
            return None

        result = parse(content)
        for w in result.warnings:
            self._warnings.append(f"{skill_md}: {w}")

        fm = result.frontmatter
        name = fm.get("name") or skill_dir.name
        description = fm.get("description", "")
        version = fm.get("version", "")
        author = fm.get("author", "")
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        refs_dir = skill_dir / "references"
        if not refs_dir.is_dir():
            refs_dir = None

        return Skill(
            name=name,
            description=description,
            version=version,
            author=author,
            tags=tags if isinstance(tags, list) else [],
            source=source,
            body=result.body,
            skill_md_path=skill_md,
            references_dir=refs_dir,
        )

    # ── 查询 ──────────────────────────────────────────────

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skills)

    def all_skills(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def enabled_skills(self) -> list[Skill]:
        disabled = self._state_store.disabled()
        return [s for s in self.all_skills() if s.name not in disabled]

    def find(self, name: str) -> Skill | None:
        skill = self._skills.get(name)
        if skill and not self._state_store.is_disabled(name):
            return skill
        return None

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def state_store(self) -> SkillStateStore:
        return self._state_store
