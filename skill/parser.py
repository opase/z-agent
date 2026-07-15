"""SKILL.md Frontmatter 解析器 — 极简 YAML 子集

不引入 PyYAML，支持 95% 实际写法：
- 单行 key: value
- 多行 key: |\n  line1\n  line2
- 行内数组 key: [a, b, c]
"""
import logging

logger = logging.getLogger(__name__)


class ParseResult:
    def __init__(self, frontmatter: dict, body: str, warnings: list[str] | None = None):
        self.frontmatter = frontmatter
        self.body = body
        self.warnings = warnings or []


def parse(full_text: str) -> ParseResult:
    if not full_text:
        return ParseResult({}, "", ["SKILL.md 内容为空"])

    normalized = full_text.replace("\r\n", "\n").replace("\r", "\n")

    if not normalized.startswith("---\n"):
        return ParseResult({}, normalized, ["缺少 frontmatter 起始标记 ---"])

    end_idx = _find_frontmatter_end(normalized)
    if end_idx < 0:
        return ParseResult({}, normalized, ["缺少 frontmatter 结束标记 ---"])

    frontmatter_text = normalized[4:end_idx]
    body = normalized[end_idx + 4:]
    if body.startswith("\n"):
        body = body[1:]

    warnings = []
    frontmatter = _parse_frontmatter(frontmatter_text, warnings)
    return ParseResult(frontmatter, body.strip(), warnings)


def _find_frontmatter_end(text: str) -> int:
    idx = 4
    while idx < len(text):
        line_end = text.find("\n", idx)
        if line_end < 0:
            return -1
        line = text[idx:line_end]
        if line == "---":
            return idx
        idx = line_end + 1
    return -1


def _parse_frontmatter(text: str, warnings: list[str]) -> dict:
    result = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue

        colon_idx = _find_key_colon(line)
        if colon_idx < 0:
            warnings.append(f"无法解析的 frontmatter 行: {line}")
            i += 1
            continue

        key = line[:colon_idx].strip()
        raw_value = line[colon_idx + 1:].strip()

        if not key:
            warnings.append(f"frontmatter 行缺少 key: {line}")
            i += 1
            continue

        if not raw_value:
            warnings.append(f"frontmatter 字段 '{key}' 缺少值")
            i += 1
            continue

        # 多行 block: key: |
        if raw_value == "|" or raw_value.startswith("|"):
            i += 1
            blocks = []
            base_indent = None
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    blocks.append("")
                    i += 1
                    continue
                indent = len(nxt) - len(nxt.lstrip(" "))
                if indent == 0:
                    break
                if base_indent is None:
                    base_indent = indent
                if indent < (base_indent or 0):
                    break
                blocks.append(nxt[(base_indent or 0):])
                i += 1
            result[key] = "\n".join(blocks).strip()
            continue

        # 行内数组: [a, b, c]
        if raw_value.startswith("[") and raw_value.endswith("]"):
            inner = raw_value[1:-1].strip()
            items = []
            if inner:
                for part in inner.split(","):
                    trimmed = part.strip().strip('"').strip("'")
                    if trimmed:
                        items.append(trimmed)
            result[key] = items
            i += 1
            continue

        # 单行字符串
        value = raw_value.strip('"').strip("'")
        result[key] = value
        i += 1

    return result


def _find_key_colon(line: str) -> int:
    in_single = False
    in_double = False
    for i, c in enumerate(line):
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == ":" and not in_single and not in_double:
            return i
    return -1
