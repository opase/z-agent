"""Skill 状态持久化 — disabled 列表"""
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillStateStore:
    """管理 disabled skill 列表，持久化到 JSON 文件"""

    def __init__(self, store_path: Path | str | None = None):
        if store_path is None:
            from config import settings as config
            store_path = Path(config.DATA_DIR) / "skills" / "state.json"
        self._path = Path(store_path)
        self._disabled: set[str] = set()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._disabled = set(data.get("disabled", []))
            except Exception:
                self._disabled = set()

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"disabled": sorted(self._disabled)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def disabled(self) -> set[str]:
        return self._disabled.copy()

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def disable(self, name: str):
        self._disabled.add(name)
        self._save()

    def enable(self, name: str):
        self._disabled.discard(name)
        self._save()
