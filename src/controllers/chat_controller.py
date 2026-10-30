
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException
from pydantic import BaseModel

from src.db import db


class CreateChatInput(BaseModel):
    title: str


class RenameChatInput(BaseModel):
    title: str


def serialize_chat(chat: dict, user_id=None) -> dict:
    return {
        "id": str(chat["_id"]),
        "title": chat["title"],
        "lastActivity": chat["lastActivity"],
        "userId": str(user_id if user_id is not None else chat["user"]),
    }


async def create_chat(data: CreateChatInput, user: dict):
    now = datetime.now(timezone.utc)

    chat_document = {
        "user": user["_id"],
        "title": data.title,
        "lastActivity": now,
        "createdAt": now,
        "updatedAt": now,
    }

    result = await db.database["chats"].insert_one(chat_document)
    chat_document["_id"] = result.inserted_id

    return {
        "message": "Chat created successfully.",
        "chat": serialize_chat(chat_document, user["_id"]),
    }


async def get_chats(user: dict):
    cursor = db.database["chats"].find(
        {"user": user["_id"]}
    ).sort("updatedAt", -1)

    chats = await cursor.to_list(length=None)

    return {
        "chats": [serialize_chat(chat) for chat in chats]
    }


async def delete_chat(chat_id: str, user: dict):
    try:
        object_id = ObjectId(chat_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat = await db.database["chats"].find_one({"_id": object_id})

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat["user"] != user["_id"]:
        raise HTTPException(
            status_code=403,
            detail="Unauthorized to delete this chat",
        )

    # Delete the chat and its messages.
    await db.database["messages"].delete_many({"chat": object_id})
    await db.database["chats"].delete_one({"_id": object_id})

    return {"message": "Chat deleted successfully"}


async def rename_chat(chat_id: str, data: RenameChatInput, user: dict):
    title = data.title.strip()

    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    try:
        object_id = ObjectId(chat_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid chat ID")

    chat = await db.database["chats"].find_one({"_id": object_id})

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat["user"] != user["_id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")

    now = datetime.now(timezone.utc)

    await db.database["chats"].update_one(
        {"_id": object_id},
        {
            "$set": {
                "title": title,
                "updatedAt": now,
            }
        },
    )

    chat["title"] = title
    chat["updatedAt"] = now

    return {
        "message": "Chat renamed successfully",
        "chat": serialize_chat(chat),
    }
