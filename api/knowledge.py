"""知识库 API"""
import logging
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Request
from core import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["知识库"])


def _get_knowledge(request: Request):
    return request.app.state.knowledge


def _get_rag(request: Request):
    return request.app.state.rag


@router.post("/upload")
async def upload(file: UploadFile = File(...), request: Request = None):
    knowledge = _get_knowledge(request)
    rag = _get_rag(request)
    ext = os.path.splitext(file.filename or "")[1].lower()

    # 统一走解析器管道（MinerU 支持所有格式，未安装时 PlainTextParser 处理 .md/.txt）
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        result = knowledge.ingest_file(tmp_path, display_name=file.filename)
    except Exception as e:
        logger.error("文件入库失败: %s", e)
        raise HTTPException(500, f"文件处理失败: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    rag.sync_bm25()
    metrics.knowledge_uploads.inc()
    metrics.vector_doc_count.set(rag.bm25.doc_count)
    return {"msg": result, "filename": file.filename}
