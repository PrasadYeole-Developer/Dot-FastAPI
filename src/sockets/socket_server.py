
import asyncio
import logging
import os
from http.cookies import SimpleCookie

import jwt
import socketio
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId

from src.db import db
from src.services.ai_service import generate_response, generate_vectors
from src.services.vector_service import create_memory, query_memory

logger = logging.getLogger(__name__)

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    cors_credentials=True,
)

# Users authenticated during the Socket.IO handshake.
connected_users: dict[str, dict] = {}


@sio.event
async def connect(sid, environ, auth=None):
    """Authenticate the connection using the JWT stored in the token cookie."""
    try:
        cookie_header = environ.get("HTTP_COOKIE", "")
        cookies = SimpleCookie()
        cookies.load(cookie_header)

        token_cookie = cookies.get("token")
        if token_cookie is None:
            logger.warning("Socket connection rejected: token cookie missing")
            return False

        secret = os.getenv("JWT_SECRET")
        if not secret:
            logger.error("JWT_SECRET is not configured")
            return False

        decoded = jwt.decode(
            token_cookie.value,
            secret,
            algorithms=["HS256"],
        )

        user_id = decoded.get("id")
        if not user_id or not ObjectId.is_valid(user_id):
            return False

        user = await db.database["users"].find_one(
            {"_id": ObjectId(user_id)}
        )
        if not user:
            return False

        connected_users[sid] = user
        logger.info("Socket user authenticated: %s", user_id)

    except (jwt.InvalidTokenError, InvalidId, ValueError):
        logger.warning("Socket connection rejected: invalid token")
        return False
    except Exception:
        logger.exception("Socket authentication failed")
        return False


@sio.event
async def disconnect(sid, reason=None):
    connected_users.pop(sid, None)


@sio.on("ai-message")
async def handle_ai_message(sid, payload):
    """Save the user's message, retrieve context, and generate an AI reply."""
    user = connected_users.get(sid)

    if not user:
        await sio.emit("error", "Unauthorized", to=sid)
        return

    if not isinstance(payload, dict):
        await sio.emit("error", "Invalid message payload", to=sid)
        return

    content = payload.get("content")
    chat_id = payload.get("chat")

    if not isinstance(content, str) or not content.strip():
        await sio.emit("error", "Message cannot be empty", to=sid)
        return

    if not isinstance(chat_id, str) or not ObjectId.is_valid(chat_id):
        await sio.emit("error", "Invalid chat ID", to=sid)
        return

    try:
        chat_object_id = ObjectId(chat_id)

        # Ensure this chat belongs to the authenticated user.
        chat = await db.database["chats"].find_one(
            {
                "_id": chat_object_id,
                "user": user["_id"],
            }
        )
        if not chat:
            await sio.emit("error", "Chat not found", to=sid)
            return

        # Save the user message and create its embedding.
        created_at = datetime.now(timezone.utc)

        message_result, vectors = await asyncio.gather(
            db.database["messages"].insert_one(
                {
                    "chat": chat_object_id,
                    "user": user["_id"],
                    "content": content.strip(),
                    "role": "user",
                    "createdAt": created_at,
                    "updatedAt": created_at,
                }
            ),
            generate_vectors(content.strip()),
        )

        message_id = str(message_result.inserted_id)
        user_id = str(user["_id"])

        # Retrieve relevant memories and the latest 20 chat messages.
        memory, history_cursor = await asyncio.gather(
            query_memory(
                query_vector=vectors,
                limit=5,
                metadata={"user": user_id},
            ),
            _get_chat_history(chat_object_id),
        )

        chat_history = history_cursor

        # Store the user's message in Pinecone.
        await create_memory(
            vectors=vectors,
            message_id=message_id,
            metadata={
                "chat": chat_id,
                "user": user_id,
                "text": content.strip(),
            },
        )

        recent_messages = [
            {
                "role": item.get("role", "user"),
                "parts": [{"text": item.get("content", "")}],
            }
            for item in chat_history
        ]

        relevant_texts = [
            match.metadata.get("text", "")
            for match in memory
            if getattr(match, "metadata", None)
            and match.metadata.get("text")
        ]

        # Keep retrieved memories separate from the conversation history.
        prompt = ""
        if relevant_texts:
            prompt += (
                "Relevant past context:\n"
                + "\n".join(relevant_texts)
                + "\n\n"
            )

        prompt += "\n".join(
            f"{item['role']}: {item['parts'][0]['text']}"
            for item in recent_messages
        )

        response = await generate_response(prompt)

        # Save the AI response and generate its embedding.
        response_created_at = datetime.now(timezone.utc)

        response_result, response_vectors = await asyncio.gather(
            db.database["messages"].insert_one(
                {
                    "chat": chat_object_id,
                    "user": user["_id"],
                    "content": response,
                    "role": "model",
                    "createdAt": response_created_at,
                    "updatedAt": response_created_at,
                }
            ),
            generate_vectors(response),
        )

        # Store the AI response in Pinecone.
        await create_memory(
            vectors=response_vectors,
            message_id=str(response_result.inserted_id),
            metadata={
                "chat": chat_id,
                "user": user_id,
                "text": response,
            },
        )

        # Emit the response only after persistence succeeds.
        await sio.emit(
            "ai-response",
            {"content": response, "chat": chat_id},
            to=sid,
        )

    except Exception:
        logger.exception("Error handling ai-message")
        await sio.emit("error", "Something went wrong", to=sid)


async def _get_chat_history(chat_object_id: ObjectId):
    cursor = (
        db.database["messages"]
        .find({"chat": chat_object_id})
        .sort("createdAt", -1)
        .limit(20)
    )
    messages = await cursor.to_list(length=20)
    messages.reverse()
    return messages
