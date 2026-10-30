
from fastapi import APIRouter, Depends

from src.controllers.message_controller import get_messages_by_chat
from src.controllers.auth_controller import get_current_user

router = APIRouter()


@router.get("/{chat_id}")
async def get_messages_route(
    chat_id: str,
    current_user: dict = Depends(get_current_user),
):
    return await get_messages_by_chat(chat_id)
