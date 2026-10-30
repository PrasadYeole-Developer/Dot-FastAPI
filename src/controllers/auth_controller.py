
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr

from src.db import db
from src.middlewares.auth_middleware import auth_user


class FullNameInput(BaseModel):
    firstName: str
    lastName: str


class RegisterInput(BaseModel):
    email: EmailStr
    fullName: FullNameInput
    password: str


class LoginInput(BaseModel):
    email: EmailStr
    password: str


def create_token(user_id: str) -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("JWT_SECRET is not configured")

    payload = {
        "id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }

    return jwt.encode(payload, secret, algorithm="HS256")


def set_auth_cookie(response: Response, token: str):
    # Preserve the existing Express cookie defaults for compatibility.
    response.set_cookie(key="token", value=token)


async def register_user(
    data: RegisterInput,
    response: Response,
):
    email = str(data.email).strip().lower()

    existing_user = await db.database["users"].find_one(
        {"email": email}
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already exists! Please try with different email.",
        )

    hashed_password = bcrypt.hashpw(
        data.password.encode("utf-8"),
        bcrypt.gensalt(rounds=10),
    ).decode("utf-8")

    now = datetime.now(timezone.utc)

    user_document = {
        "email": email,
        "fullName": {
            "firstName": data.fullName.firstName,
            "lastName": data.fullName.lastName,
        },
        "password": hashed_password,
        "createdAt": now,
        "updatedAt": now,
    }

    result = await db.database["users"].insert_one(user_document)
    user_id = str(result.inserted_id)

    token = create_token(user_id)
    set_auth_cookie(response, token)

    return {
        "message": "User registered successfully.",
        "user": {
            "id": user_id,
            "email": email,
            "name": user_document["fullName"],
        },
    }


async def login_user(
    data: LoginInput,
    response: Response,
):
    email = str(data.email).strip().lower()

    user = await db.database["users"].find_one(
        {"email": email}
    )

    if not user:
        raise HTTPException(
            status_code=400,
            detail="Invalid email! Please enter correct email id.",
        )

    stored_password = user.get("password")

    if not stored_password or not bcrypt.checkpw(
        data.password.encode("utf-8"),
        stored_password.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid password! Please enter correct password.",
        )

    token = create_token(str(user["_id"]))
    set_auth_cookie(response, token)

    return {
        "message": "User logged in successfully.",
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["fullName"],
        },
    }


async def logout_user(response: Response):
    response.delete_cookie(key="token")

    return {
        "message": "User logged out successfully.",
    }


async def get_current_user(
    current_user: dict = Depends(auth_user),
):
    return {
        "user": {
            "id": str(current_user["_id"]),
            "email": current_user["email"],
            "name": current_user["fullName"],
        }
    }
