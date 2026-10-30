
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException

from src.db import db


async def get_messages_by_chat(chat_id: str):
    try:
        object_id = ObjectId(chat_id)

        messages = await db.database["messages"].find(
            {"chat": object_id}
        ).sort("createdAt", 1).to_list(length=None)

        for message in messages:
            message["id"] = str(message.pop("_id"))
            message["user"] = (
                str(message["user"])
                if message.get("user") is not None
                else None
            )
            message["chat"] = (
                str(message["chat"])
                if message.get("chat") is not None
                else None
            )

        return {"messages": messages}

    except (InvalidId, TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid chat ID",
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch messages",
        )
