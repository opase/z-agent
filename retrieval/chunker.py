"""
结构感知父子分块器

父块: 按标题→段落→句子→标点→硬切 5 级降级，目标 500-1500 字符
子块: 父块内从句子边界开始，~250 tokens (~375 字符)，20% 重叠，不跨父块
"""
from __future__ import annotations
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── 拆分正则 ──
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
_PARAGRAPH_SEP_RE = re.compile(r'\n{2,}')     # 段落边界（空行）
_SENTENCE_SEP_RE = re.compile(r'(?<=[。！？])\s*')   # 句子边界
_PUNCT_SEP_RE = re.compile(r'(?<=[；，,;])\s*')       # 标点边界

# 页面标记（MinerU 常见格式）
_PAGE_MARKER_RE = re.compile(
    r'<!--\s*PAGE\s+(\d+)\s*-->|\[PAGE\s+(\d+)\]|^\s*#+\s*Page\s+(\d+)\s*$',
    re.MULTILINE | re.IGNORECASE,
)

# ── 配置常量 ──
PARENT_MIN_CHARS = 500       # 父块最小字符
PARENT_MAX_CHARS = 1500      # 父块最大字符（超此触发降级）
CHILD_TARGET_TOKENS = 250    # 子块目标 token
CHILD_OVERLAP_RATIO = 0.2    # 子块重叠比例

# 中文约 1.5 字符/token，但实际差异大，用更保守的 1.2
_CHARS_PER_TOKEN = 1.2
CHILD_TARGET_CHARS = int(CHILD_TARGET_TOKENS * _CHARS_PER_TOKEN)  # ~300


@dataclass
class Chunk:
    """分块数据"""
    id: str
    text: str
    is_parent: bool                  # True=父块(送LLM), False=子块(检索)
    parent_id: Optional[str] = None
    page: Optional[int] = None
    page_end: Optional[int] = None
    section: str = ""               # 标题路径，如 "安装 > 环境要求"
    heading_level: int = 0
    chunk_type: str = "text"


def _make_id(text: str, prefix: str, index: int) -> str:
    h = hashlib.md5(text.encode()[:200]).hexdigest()[:8]
    return f"{prefix}_{index}_{h}"


def _extract_page(text: str) -> Optional[int]:
    m = _PAGE_MARKER_RE.search(text)
    if m:
        for g in m.groups():
            if g and g.isdigit():
                return int(g)
    return None


def _strip_page_markers(text: str) -> str:
    return _PAGE_MARKER_RE.sub("", text)


class StructuredChunker:
    """结构感知父子分块器

    父块: 5 级降级 (标题→段落→句子→标点→硬切)，500-1500 字符
    子块: 句子边界开始，滑动窗口 20% 重叠，~300 字符
    """

    def __init__(self):
        self._counter = 0

    # ── 入口 ──

    def chunk(self, text: str, source_name: str = "") -> tuple[list[Chunk], list[Chunk]]:
        """执行父子分块

        Returns:
            (parent_chunks, child_chunks): 父块列表和子块列表，
            子块通过 parent_id 引用父块。
        """
        if not text.strip():
            return [], []

        self._counter = 0
        all_parents: list[Chunk] = []
        all_children: list[Chunk] = []

        # ── Step 1: 按标题拆大段 ──
        sections = self._split_by_headings(text) or [{
            "heading": "", "level": 0, "body": text, "section_path": ""
        }]

        current_page = 1

        for sec in sections:
            sec_page = _extract_page(sec["body"])
            if sec_page:
                current_page = sec_page

            # ── Step 2: 父块拆分（5 级降级）──
            parents = self._split_parents(
                sec["body"], section=sec["section_path"],
                heading_level=sec["level"], start_page=current_page,
            )

            # 更新页码
            for p in parents:
                if p.page:
                    current_page = p.page
                if not p.page:
                    p.page = current_page

                all_parents.append(p)

                # ── Step 3: 父块内拆子块（句子边界 + 滑动窗口 + 20% 重叠）──
                children = self._split_children(p)
                all_children.extend(children)

        # 挂载 source
        for c in all_parents + all_children:
            c.metadata = {"source": source_name}

        logger.info(
            "分块完成: %s → parents=%d children=%d",
            source_name, len(all_parents), len(all_children),
        )
        return all_parents, all_children

    # ── 标题拆分 ──

    def _split_by_headings(self, text: str) -> list[dict]:
        heading_positions = [
            (m.start(), m.group(1), m.group(2).strip())
            for m in _HEADING_RE.finditer(text)
        ]
        if not heading_positions:
            return []

        sections = []
        heading_stack: list[str] = []

        for i, (pos, hashes, title) in enumerate(heading_positions):
            level = len(hashes)
            heading_line_end = (
                text.index("\n", pos) if "\n" in text[pos:]
                else pos + len(hashes) + 1 + len(title)
            )
            body_end = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(text)
            body = text[heading_line_end:body_end].strip()

            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            section_path = " > ".join(heading_stack)

            sections.append({
                "heading": title, "level": level,
                "body": body, "section_path": section_path,
            })
        return sections

    # ── 父块拆分: 5 级降级 ──

    def _split_parents(
        self, text: str, section: str = "", heading_level: int = 0,
        start_page: int = 1,
    ) -> list[Chunk]:
        """按 5 级降级策略拆分为父块"""
        text = text.strip()
        if not text:
            return []

        current_page = start_page
        result: list[Chunk] = []

        # 先清除页面标记再记录
        cleaned = _strip_page_markers(text)

        # 如果文本在目标范围内，直接作为一个父块
        if len(cleaned) <= PARENT_MAX_CHARS and len(cleaned) >= PARENT_MIN_CHARS:
            page = _extract_page(text) or current_page
            result.append(self._make_parent(cleaned, section, heading_level, page))

        elif len(cleaned) < PARENT_MIN_CHARS:
            # 过短：直接作为父块（后面 merge_small 会处理合并）
            page = _extract_page(text) or current_page
            result.append(self._make_parent(cleaned, section, heading_level, page))

        else:
            # 超过 PARENT_MAX_CHARS，逐级降级拆分
            result = self._degrade_split(cleaned, section, heading_level, current_page)

        return result

    def _degrade_split(
        self, text: str, section: str, heading_level: int, start_page: int,
    ) -> list[Chunk]:
        """5 级降级拆分，每级尽量产生 500-1500 字符的块"""
        current_page = start_page

        # 尝试各级拆分，找到产生合适大小块的策略
        strategies = [
            ("paragraph", _PARAGRAPH_SEP_RE),
            ("sentence", _SENTENCE_SEP_RE),
            ("punctuation", _PUNCT_SEP_RE),
            ("character", None),
        ]

        for name, sep_re in strategies:
            if sep_re:
                parts = [p.strip() for p in sep_re.split(text) if p.strip()]
            else:
                parts = [text]

            # 合并过小的部分
            parts = self._merge_small_parts(parts)

            # 检查是否所有部分都 ≤ PARENT_MAX_CHARS
            if all(len(p) <= PARENT_MAX_CHARS for p in parts):
                result = []
                for p_text in parts:
                    page = _extract_page(p_text) or current_page
                    result.append(self._make_parent(p_text, section, heading_level, page))
                    if page != current_page:
                        current_page = page
                return result

            # 还有超限的 → 降到下一级
            logger.debug("父块拆分: %s 级有超限块，降级", name)
            # 将超限部分展开再试下一级
            new_texts = []
            for p in parts:
                if len(p) > PARENT_MAX_CHARS:
                    # 超限的部分用下一级策略继续拆
                    if name == "paragraph":
                        new_texts.extend(
                            s.strip() for s in _SENTENCE_SEP_RE.split(p) if s.strip()
                        )
                    elif name == "sentence":
                        new_texts.extend(
                            s.strip() for s in _PUNCT_SEP_RE.split(p) if s.strip()
                        )
                    else:
                        new_texts.append(p)
                else:
                    new_texts.append(p)
            # 让循环继续，带新 parts 重新判断
            text = "\n\n".join(new_texts)  # 重新合并，下一次迭代会重新 split

        # 最后兜底：字符数硬切
        return self._hard_split(text, section, heading_level, start_page)

    def _hard_split(
        self, text: str, section: str, heading_level: int, start_page: int,
    ) -> list[Chunk]:
        """字符数硬切（优先级 5）"""
        result = []
        current_page = start_page
        pos = 0
        while pos < len(text):
            end = min(pos + PARENT_MAX_CHARS, len(text))
            chunk_text = text[pos:end].strip()
            if chunk_text:
                page = _extract_page(chunk_text) or current_page
                result.append(self._make_parent(chunk_text, section, heading_level, page))
                if page != current_page:
                    current_page = page
            pos = end
        return result

    def _merge_small_parts(self, parts: list[str]) -> list[str]:
        """合并过小的文本片段，使每个片段不小于 PARENT_MIN_CHARS/2"""
        if not parts:
            return parts
        threshold = PARENT_MIN_CHARS // 2
        merged: list[str] = []
        buffer: list[str] = []

        for p in parts:
            buffer.append(p)
            total = sum(len(b) for b in buffer)
            if total >= threshold and total <= PARENT_MAX_CHARS:
                merged.append(" ".join(buffer))
                buffer = []
            elif total > PARENT_MAX_CHARS:
                # 超了，先提交已有 buffer（不含当前）
                if len(buffer) > 1:
                    merged.append(" ".join(buffer[:-1]))
                    buffer = [buffer[-1]]
                else:
                    merged.append(p)
                    buffer = []

        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + " ".join(buffer)
            else:
                merged.append(" ".join(buffer))

        return merged

    # ── 子块拆分: 句子边界 + 滑动窗口 + 20% 重叠 ──

    def _split_children(self, parent: Chunk) -> list[Chunk]:
        """从父块切出重叠子块"""
        text = parent.text
        if not text.strip():
            return []

        # 先按句子切
        sentences = [s.strip() for s in _SENTENCE_SEP_RE.split(text) if s.strip()]
        if not sentences:
            return []

        children: list[Chunk] = []
        # 滑动窗口，目标 CHILD_TARGET_CHARS，重叠 CHILD_OVERLAP_RATIO
        overlap_chars = int(CHILD_TARGET_CHARS * CHILD_OVERLAP_RATIO)  # ~60

        i = 0
        while i < len(sentences):
            window: list[str] = []
            window_len = 0

            # 从位置 i 开始，尽可能填满窗口
            j = i
            while j < len(sentences) and window_len < CHILD_TARGET_CHARS * 1.3:
                s = sentences[j]
                window.append(s)
                window_len += len(s)
                j += 1

            if not window:
                i += 1
                continue

            child_text = " ".join(window).strip()
            if child_text:
                child = self._make_child(child_text, parent)
                children.append(child)

            # 如果窗口长度不足目标（最后一个窗口），结束
            if j >= len(sentences):
                break

            # 计算下一个窗口起始位置（回退 overlap 量）
            # 找到累计字符数超过 window_len - overlap_chars 的句子位置
            cumulative = 0
            next_i = i
            for k in range(i, j):
                cumulative += len(sentences[k])
                if cumulative >= window_len - overlap_chars:
                    next_i = k + 1
                    break
            i = max(next_i, i + 1)

        return children

    # ── 工厂方法 ──

    def _make_parent(
        self, text: str, section: str, heading_level: int, page: int,
    ) -> Chunk:
        self._counter += 1
        return Chunk(
            id=_make_id(text, "P", self._counter),
            text=text,
            is_parent=True,
            page=page,
            section=section,
            heading_level=heading_level,
        )

    def _make_child(self, text: str, parent: Chunk) -> Chunk:
        self._counter += 1
        return Chunk(
            id=_make_id(text, "C", self._counter),
            text=text,
            is_parent=False,
            parent_id=parent.id,
            page=parent.page,
            section=parent.section,
            heading_level=parent.heading_level,
        )
