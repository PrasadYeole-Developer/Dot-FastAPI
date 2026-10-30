
from fastapi import APIRouter, Depends

from src.controllers.chat_controller import (
    CreateChatInput,
    RenameChatInput,
    create_chat,
    get_chats,
    delete_chat,
    rename_chat,
)
from src.middlewares.auth_middleware import auth_user

router = APIRouter()


@router.post("/create-chat", status_code=201)
async def create_chat_route(
    data: CreateChatInput,
    user: dict = Depends(auth_user),
):
    return await create_chat(data, user)


@router.get("/")
async def get_chats_route(
    user: dict = Depends(auth_user),
):
    return await get_chats(user)


@router.delete("/{chat_id}")
async def delete_chat_route(
    chat_id: str,
    user: dict = Depends(auth_user),
):
    return await delete_chat(chat_id, user)


@router.patch("/{chat_id}")
async def rename_chat_route(
    chat_id: str,
    data: RenameChatInput,
    user: dict = Depends(auth_user),
):
    return await rename_chat(chat_id, data, user)
