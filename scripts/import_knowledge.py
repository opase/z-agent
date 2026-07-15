"""
批量导入知识库文档到 RAG 系统
从 config.KNOWLEDGE_DIR（data/knowledge）读取全部文档并入库。
自动检测 MinerU：可用时解析 PDF/DOCX/PPTX，不可用时仅导入纯文本。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings as config
from core.knowledge_service import KnowledgeService
from core.rag_service import RagService

# 支持的扩展名（纯文本 + MinerU 格式）
_TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json"}
_MINERU_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}


def import_knowledge():
    kb = KnowledgeService()
    rag = RagService()

    knowledge_dir = config.KNOWLEDGE_DIR
    if not os.path.exists(knowledge_dir):
        print(f"❌ {knowledge_dir} 目录不存在")
        return

    # 自动检测 MinerU
    from retrieval.document_parser import MinerUParser
    has_mineru = MinerUParser.is_available()

    all_exts = _TEXT_EXTS | _MINERU_EXTS if has_mineru else _TEXT_EXTS
    files = [f for f in os.listdir(knowledge_dir)
             if os.path.splitext(f)[1].lower() in all_exts]
    print(f"找到 {len(files)} 个知识文档 (MinerU: {'✅' if has_mineru else '未安装'})")

    for filename in files:
        filepath = os.path.join(knowledge_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in _MINERU_EXTS or ext in _TEXT_EXTS:
            # 通过 ingest_file 走新流水线（清洗 + 父子分块 + 语义去重）
            result = kb.ingest_file(filepath, display_name=filename)
        else:
            # 兼容旧的纯文本路径
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            result = kb.upload(content, filename)

        print(f"  {filename}: {result}")

    print("\n知识库导入完成！")

    rag.sync_bm25()
    print("BM25 索引同步完成！")


if __name__ == "__main__":
    import_knowledge()
