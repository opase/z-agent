"""对话 API"""
import base64
import json
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from config import settings as config

router = APIRouter(prefix="/chat", tags=["对话"])


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    user_id: str = "default"
    mode: str = "auto"  # "auto" | "react" | "plan" | "multi_agent"


def _get_rag(request: Request):
    return request.app.state.rag


def _get_sessions(request: Request):
    return request.app.state.sessions


def _validate_images(files: list[UploadFile]) -> list[str]:
    """校验图片并转为 base64 列表"""
    if len(files) > config.max_images_per_message:
        raise HTTPException(400, f"单次最多上传 {config.max_images_per_message} 张图片")
    result = []
    max_bytes = config.max_image_size_mb * 1024 * 1024
    for f in files:
        data = f.file.read()
        if len(data) > max_bytes:
            raise HTTPException(400, f"图片 {f.filename} 超过 {config.max_image_size_mb}MB 限制")
        b64 = base64.b64encode(data).decode("utf-8")
        # 拼接 data URI 前缀，DashScope 多模态 API 需要
        content_type = f.content_type or "image/jpeg"
        result.append(f"data:{content_type};base64,{b64}")
    return result


# ==================== 纯文本接口（保持不变，向后兼容）====================

@router.post("")
async def chat(req: ChatRequest, request: Request):
    rag = _get_rag(request)
    sessions = _get_sessions(request)
    session = sessions.get_or_create(req.session_id, req.user_id)
    result = await rag.chat(req.question, session.memory, req.user_id, mode=req.mode)

    # HITL: 检查是否被审批中断
    if result.get("status") == "interrupted":
        return {
            "status": "interrupted",
            "interrupt": result["interrupt"],
            "session_id": session.session_id,
        }

    return {
        "answer": result["answer"], "session_id": session.session_id,
        "turn_count": session.memory.turn_count, "mode": result.get("mode", ""),
        "verification": result.get("verification", {}),
    }


@router.post("/stream")
async def chat_stream(req: ChatRequest, request: Request):
    rag = _get_rag(request)
    sessions = _get_sessions(request)
    session = sessions.get_or_create(req.session_id, req.user_id)

    media_type = "text/event-stream" if config.sse_structured_events else "text/plain"

    async def generate():
        meta_accumulated = None
        async for raw in rag.chat_stream(
            req.question, session.memory, req.user_id,
            session_id=session.session_id, mode=req.mode,
        ):
            if config.sse_structured_events:
                # 检测是否为元数据标记（旧格式 __CA_META__ 兼容）
                if isinstance(raw, str) and raw.startswith("\n__CA_META__"):
                    # 收集 meta 用于 done 事件
                    import re
                    match = re.search(r"__CA_META__(\{.*?\})__CA_META_END__", raw, re.DOTALL)
                    if match:
                        try:
                            meta_accumulated = json.loads(match[1])
                        except json.JSONDecodeError:
                            pass
                    continue

                # 结构化 SSE 事件格式
                if isinstance(raw, str) and not raw.startswith("data:"):
                    yield f"data: {json.dumps({'type': 'token', 'content': raw}, ensure_ascii=False)}\n\n"
                else:
                    yield raw
            else:
                # 旧格式：纯文本 token
                yield raw

        # 发送 done 事件（替代 __CA_META__ 标记）
        if config.sse_structured_events:
            # 如果没有收集到 meta，构建默认值
            if not meta_accumulated:
                meta_accumulated = {
                    "session_id": session.session_id,
                    "turn_count": session.memory.turn_count if session.memory else 1,
                }
            yield f"data: {json.dumps({'type': 'done', 'meta': meta_accumulated}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type=media_type)


# ==================== 图片对话接口（multipart/form-data）====================

@router.post("/image")
async def chat_image(
    question: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Form("default"),
    images: list[UploadFile] = File(default=[]),
    request: Request = None,
):
    rag = _get_rag(request)
    sessions = _get_sessions(request)

    # 校验并转 base64
    image_b64_list = _validate_images(images) if images else []

    session = sessions.get_or_create(session_id, user_id)
    result = await rag.chat(question, session.memory, user_id, images=image_b64_list)
    return {
        "answer": result["answer"], "session_id": session.session_id,
        "turn_count": session.memory.turn_count, "mode": result.get("mode", ""),
        "verification": result.get("verification", {}),
        # 返回图像识别结果给前端
        "image_desc": result.get("image_desc", ""),
        "detected_products": result.get("detected_products", []),
    }


@router.post("/image/stream")
async def chat_image_stream(
    question: str = Form(...),
    session_id: str | None = Form(None),
    user_id: str = Form("default"),
    images: list[UploadFile] = File(default=[]),
    request: Request = None,
):
    rag = _get_rag(request)
    sessions = _get_sessions(request)

    image_b64_list = _validate_images(images) if images else []
    session = sessions.get_or_create(session_id, user_id)

    async def generate():
        async for token in rag.chat_stream(
            question, session.memory, user_id, images=image_b64_list,
            session_id=session.session_id,
        ):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ==================== 会话管理 ====================

@router.post("/{session_id}/end")
async def end_session(session_id: str, request: Request, user_id: str = "default"):
    rag = _get_rag(request)
    sessions = _get_sessions(request)
    session = sessions.get(session_id, user_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    await rag.end_session(user_id, session.memory)
    await sessions.safe_remove(session_id)  # Phase 3: 等待压缩任务完成
    return {"msg": f"会话 {session_id} 已结束"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request):
    """删除会话（包括 0 轮会话）"""
    sessions = _get_sessions(request)
    sessions.delete(session_id)
    return {"msg": f"会话 {session_id} 已删除"}


@router.get("/{session_id}/history")
async def get_history(
    session_id: str, request: Request, user_id: str = "default",
    limit: int = 0, offset: int = 0,
):
    sessions = _get_sessions(request)
    session = sessions.get(session_id, user_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    # 从 SQLite 取完整历史，不受 ConversationMemory 窗口限制
    all_msgs = sessions.store.get_messages(session_id)
    if limit > 0:
        all_msgs = all_msgs[offset:offset + limit]
    elif offset > 0:
        all_msgs = all_msgs[offset:]
    full_messages = [
        {"role": m["role"], "content": m["content"],
         "image_count": m.get("image_count", 0),
         "timestamp": m.get("timestamp", "")}
        for m in all_msgs
    ]
    return {"session_id": session_id, "messages": full_messages,
            "total": len(all_msgs),
            "summary": session.memory.summary}
