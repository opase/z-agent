"""用户 API"""
from fastapi import APIRouter, Request
from memory.long_term import LongTermMemory

router = APIRouter(prefix="/users", tags=["用户"])


def _get_sessions(request: Request):
    return request.app.state.sessions


@router.get("/{user_id}/profile")
async def get_profile(user_id: str):
    ltm = LongTermMemory(user_id)
    return {"user_id": user_id, "data": ltm.data}


@router.get("/sessions")
async def list_sessions(request: Request):
    sessions = _get_sessions(request)
    return {"sessions": sessions.list_all()}
