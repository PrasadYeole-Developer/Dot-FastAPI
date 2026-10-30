
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class FullName(BaseModel):
    firstName: str
    lastName: str


class UserModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    email: EmailStr
    fullName: FullName
    password: str | None = None

    createdAt: datetime | None = None
    updatedAt: datetime | None = None
