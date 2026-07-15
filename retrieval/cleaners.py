"""
文档噪声过滤流水线

可插拔 Cleaner 策略，串行执行。默认启用基础过滤，可通过 config 开关。
"""
from __future__ import annotations
import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Cleaner(ABC):
    """噪声过滤器抽象基类"""

    @abstractmethod
    def clean(self, text: str) -> str:
        """过滤噪声，返回清洗后文本"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """过滤器名称"""
        ...


class PageNumberCleaner(Cleaner):
    """移除独立页码行"""
    name = "page_number"

    _PATTERNS = [
        re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE),           # 纯数字行
        re.compile(r'^\s*第\s*\d{1,4}\s*页\s*$', re.MULTILINE),  # "第X页"
        re.compile(r'^\s*-\s*\d{1,4}\s*-\s*$', re.MULTILINE),    # "- 5 -"
    ]

    def clean(self, text: str) -> str:
        for pat in self._PATTERNS:
            text = pat.sub("", text)
        return text


class ShortLineCleaner(Cleaner):
    """移除无意义短噪声行"""
    name = "short_line"

    _GARBAGE_RE = re.compile(
        r'^\s*[|·•·˙¨˚ˉˊˋ˘˛˝˞˟ˠˡˢˣˤ˥˦˧˨˩˪˫ˬ˭ˮ˯˰˱˲˳˴'
        r'\x00-\x08\x0b\x0c\x0e-\x1f]+\s*$', re.MULTILINE
    )

    def clean(self, text: str) -> str:
        # 移除纯符号行（< 3 个有意义字符）
        lines = text.split("\n")
        kept = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            # 纯符号/乱码行
            if self._GARBAGE_RE.match(stripped):
                continue
            # 有效中英文字符计数
            meaningful = sum(1 for c in stripped if c.isalnum() or '一' <= c <= '鿿')
            if len(stripped) <= 3 and meaningful == 0:
                continue
            kept.append(line)
        return "\n".join(kept)


class BoilerplateCleaner(Cleaner):
    """移除常见文档模板文本（版权声明、免责声明等）"""
    name = "boilerplate"

    _BOILERPLATE_PATTERNS = [
        re.compile(r'Copyright\s*[©⊂]\s*\d{4}.*', re.IGNORECASE),
        re.compile(r'All\s+Rights\s+Reserved\.?', re.IGNORECASE),
        re.compile(r'Confidential|Proprietary|机密|内部资料', re.IGNORECASE),
        re.compile(r'未经.*许可.*不得.*(复制|转载|传播)', re.IGNORECASE),
        re.compile(r'本文档.*最终解释权.*', re.IGNORECASE),
    ]

    def clean(self, text: str) -> str:
        for pat in self._BOILERPLATE_PATTERNS:
            text = pat.sub("", text)
        return text


class HeaderFooterCleaner(Cleaner):
    """移除页眉页脚（文档内重复出现 ≥3 次的行）"""
    name = "header_footer"

    def clean(self, text: str) -> str:
        lines = text.split("\n")
        stripped = [l.strip() for l in lines]

        # 统计每行出现次数（忽略空行和短行）
        freq: dict[str, int] = {}
        for s in stripped:
            if len(s) >= 5 and len(s) <= 80:
                freq[s] = freq.get(s, 0) + 1

        # 出现 ≥3 次的行视为页眉/页脚
        repeated = {s for s, c in freq.items() if c >= 3}

        if repeated:
            lines = [l for l in lines if l.strip() not in repeated]

        return "\n".join(lines)


class CleaningPipeline:
    """噪声过滤流水线

    Args:
        cleaners: 过滤器列表，按顺序执行

    用法:
        pipeline = CleaningPipeline.default()
        clean_text = pipeline.clean(raw_text)
    """

    def __init__(self, cleaners: list[Cleaner] = None):
        self._cleaners = cleaners or []

    @classmethod
    def default(cls) -> "CleaningPipeline":
        """默认流水线：页码 → 短噪声 → 模板文本 → 页眉页脚"""
        return cls([
            PageNumberCleaner(),
            ShortLineCleaner(),
            BoilerplateCleaner(),
            HeaderFooterCleaner(),
        ])

    @classmethod
    def minimal(cls) -> "CleaningPipeline":
        """最小流水线：仅页码"""
        return cls([PageNumberCleaner()])

    @classmethod
    def none(cls) -> "CleaningPipeline":
        """空流水线：不做任何清洗"""
        return cls([])

    def clean(self, text: str) -> str:
        for cleaner in self._cleaners:
            before = len(text)
            text = cleaner.clean(text)
            after = len(text)
            if before != after:
                logger.debug("清洗器 %s: %d → %d 字符 (-%d)",
                             cleaner.name, before, after, before - after)
        return text
