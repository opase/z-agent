"""
文档解析器 — 可插拔设计

默认 PlainTextParser：直接读取 .txt/.md（零依赖）。
可选 MinerUParser：PDF/DOCX/PPTX → markdown（需 pip install mineru[all]）。

通过配置 DOCUMENT_PARSER=auto 自动检测（MinerU 可用则用，否则降级纯文本）。
"""
from __future__ import annotations
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# MinerU 支持的格式
_MINERU_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}


class DocumentParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        """解析文档，返回纯文本（markdown 格式）"""
        ...

    @abstractmethod
    def supports(self, extension: str) -> bool:
        """是否支持该文件扩展名（含点，如 '.pdf'）"""
        ...


class PlainTextParser(DocumentParser):
    """纯文本解析器 — 直接读取 .txt/.md

    这是默认解析器，零额外依赖。非文本格式直接拒绝。
    """

    def parse(self, file_path: str) -> str:
        ext = _ext(file_path)
        if ext not in (".txt", ".md", ".markdown", ".csv", ".json"):
            raise ValueError(f"PlainTextParser 不支持格式: {ext}（仅 .txt/.md）")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def supports(self, extension: str) -> bool:
        return extension in (".txt", ".md", ".markdown", ".csv", ".json")


class MinerUParser(DocumentParser):
    """MinerU 解析器 — 支持本地 CLI 和云端 API 两种模式

    本地模式（默认）:
        前置条件: uv pip install "mineru[all]"
        调用方式: subprocess 执行 mineru CLI

    云端模式:
        前置条件: pip install mineru-open-sdk
        调用方式: Python SDK → mineru.net API
        无需 GPU，零环境配置，数据出网

        两种子模式:
        - flash: 免 Token，≤10MB/≤20页，仅输出 Markdown
        - precision: 需免费 Token，≤200MB/≤600页，支持批量 + 多格式输出
    """

    def __init__(self, method: str = "auto", api_token: str = None):
        """
        Args:
            method: "auto" | "cli" | "api-flash" | "api-precision"
                - auto: 优先本地 CLI → 降级云端 Flash
            api_token: 云端精准模式 token（从 mineru.net/apiManage/token 免费获取）
        """
        self._method = method
        self._api_token = api_token

        if method == "cli":
            self._check_cli()
        elif method in ("api-flash", "api-precision"):
            self._check_api_sdk()
        # auto: 延迟检查，parse() 时再决定

    # ── 可用性检测 ──

    @staticmethod
    def is_cli_available() -> bool:
        """检测本地 mineru CLI 是否可用"""
        try:
            subprocess.run(["mineru", "--version"], capture_output=True, timeout=10)
            return True
        except Exception:
            return False

    @staticmethod
    def is_api_available() -> bool:
        """检测云端 SDK 是否可用"""
        try:
            import mineru  # mineru-open-sdk
            return True
        except ImportError:
            return False

    def _check_cli(self):
        if not self.is_cli_available():
            raise RuntimeError(
                "MinerU CLI 不可用。请先安装: uv pip install \"mineru[all]\""
            )

    def _check_api_sdk(self):
        if not self.is_api_available():
            raise RuntimeError(
                "MinerU SDK 不可用。请先安装: pip install mineru-open-sdk"
            )

    # ── 解析入口 ──

    def parse(self, file_path: str) -> str:
        ext = _ext(file_path)
        if ext not in _MINERU_EXTENSIONS and ext not in (".txt", ".md", ".markdown", ".csv", ".json"):
            raise ValueError(f"MinerUParser 不支持的格式: {ext}")

        # 自动选择模式
        method = self._method
        if method == "auto":
            if self.is_cli_available():
                method = "cli"
            elif self.is_api_available():
                method = "api-flash"
                logger.info("MinerU CLI 不可用，降级云端 Flash 模式")
            else:
                raise RuntimeError(
                    "MinerU 不可用。请安装本地版: uv pip install \"mineru[all]\"，"
                    "或云端版: pip install mineru-open-sdk"
                )

        if method == "cli":
            return self._parse_cli(file_path)
        if method in ("api-flash", "api-precision"):
            return self._parse_api(file_path, method)
        raise ValueError(f"未知 MinerU 模式: {method}")

    # ── 本地 CLI ──

    def _parse_cli(self, file_path: str) -> str:
        with tempfile.TemporaryDirectory() as out_dir:
            subprocess.run(
                ["mineru", "-p", file_path, "-o", out_dir],
                check=True, capture_output=True, text=True,
            )
            md_path = self._find_md(out_dir, file_path)
            with open(md_path, "r", encoding="utf-8") as f:
                return self._clean(f.read())

    # ── 云端 API ──

    def _parse_api(self, file_path: str, method: str) -> str:
        import mineru as mineru_sdk

        try:
            if method == "api-flash":
                client = mineru_sdk.MinerU()  # 免 Token
                result = client.flash_extract(file_path)
            else:
                token = self._api_token or os.getenv("MINERU_API_TOKEN", "")
                if not token:
                    raise RuntimeError(
                        "云端精准模式需要 Token。"
                        "设置方式: export MINERU_API_TOKEN=your-token"
                    )
                client = mineru_sdk.MinerU(token)
                result = client.extract(file_path)
        except Exception as e:
            # MinerU SDK 内部可能抛各种异常（API 业务错误、网络超时等），
            # 转为 RuntimeError 带原始信息，让上层统一处理。
            raise RuntimeError(f"MinerU 云端解析失败: {e}") from e

        # 提取 markdown 内容
        if hasattr(result, "markdown"):
            return self._clean(result.markdown)
        if hasattr(result, "content"):
            return self._clean(result.content)
        return self._clean(str(result))

    # ── 工具方法 ──

    @staticmethod
    def _find_md(out_dir: str, file_path: str) -> str:
        base = os.path.splitext(os.path.basename(file_path))[0]
        md_path = os.path.join(out_dir, base, f"{base}.md")
        if os.path.exists(md_path):
            return md_path
        candidates = []
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.endswith(".md"):
                    candidates.append(os.path.join(root, f))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"MinerU 未产出 markdown 文件: {out_dir}")

    @staticmethod
    def is_available() -> bool:
        """检测 MinerU 是否可用（本地 CLI 或云端 SDK 任一即可）"""
        return MinerUParser.is_cli_available() or MinerUParser.is_api_available()

    def supports(self, extension: str) -> bool:
        return extension in _MINERU_EXTENSIONS or extension in (".txt", ".md", ".markdown")

    @staticmethod
    def _clean(md_text: str) -> str:
        """移除 MinerU 元信息标记（IMAGE/TABLE），保留 PAGE 标记供分块器使用"""
        import re
        # 移除 IMAGE / TABLE 标记，但保留 PAGE 标记
        cleaned = re.sub(r"<!--\s*(IMAGE|TABLE).*?-->", "", md_text)
        # 将 MinerU 的 page 标记统一为 <!-- PAGE N --> 格式（供 chunker 识别）
        cleaned = re.sub(
            r"\[PAGE\s+(\d+)\]",
            r"<!-- PAGE \1 -->",
            cleaned, flags=re.IGNORECASE,
        )
        # 移除多余空行但不破坏表格
        cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
        return cleaned.strip()


# ── 工厂 / 自动检测 ──

def create_parser(preference: str = "auto") -> DocumentParser:
    """创建文档解析器

    Args:
        preference:
            - "auto":   MinerU 可用则用（CLI > 云端 Flash），不可用降级 PlainTextParser
            - "plain":  始终使用纯文本解析器
            - "mineru": 强制 MinerU（方法由 MINERU_METHOD 环境变量控制）

    环境变量:
        MINERU_METHOD:      cli | api-flash | api-precision（默认 auto）
        MINERU_API_TOKEN:  云端精准模式 token
    """
    if preference == "plain":
        logger.info("文档解析器: PlainTextParser（纯文本模式）")
        return PlainTextParser()

    if preference in ("mineru", "auto"):
        mineru_method = os.getenv("MINERU_METHOD", "auto")
        api_token = os.getenv("MINERU_API_TOKEN", None)

        if preference == "mineru":
            # 强制 MinerU
            logger.info("文档解析器: MinerUParser（强制, method=%s）", mineru_method)
            return MinerUParser(method=mineru_method, api_token=api_token)

        # auto: 优先 MinerU，不可用降级纯文本
        if MinerUParser.is_available():
            logger.info("文档解析器: MinerUParser（自动检测, method=%s）", mineru_method)
            return MinerUParser(method=mineru_method, api_token=api_token)

        logger.info("文档解析器: PlainTextParser（MinerU 未安装，降级纯文本）")
        return PlainTextParser()

    # 直接传 method（如 "api-flash"）
    logger.info("文档解析器: MinerUParser（method=%s）", preference)
    return MinerUParser(method=preference)


def _ext(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()
