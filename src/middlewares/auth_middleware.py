
import os

import jwt
from bson import ObjectId
from fastapi import Cookie, HTTPException, Request

from src.db import db


async def auth_user(
    request: Request,
    token: str | None = Cookie(default=None),
):
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )

    try:
        secret = os.getenv("JWT_SECRET")
        if not secret:
            raise RuntimeError("JWT_SECRET is not configured")

        decoded = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
        )

        user_id = decoded.get("id")

        if not user_id or not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
            )

        user = await db.database["users"].find_one(
            {"_id": ObjectId(user_id)}
        )

        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found",
            )

        request.state.user = user
        return user

    except HTTPException:
        raise
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )
