
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MessageModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    user: str | None = None
    chat: str | None = None
    content: str
    role: str = Field(
        default="user",
        pattern="^(user|model|system)$",
    )

    createdAt: datetime | None = None
    updatedAt: datetime | None = None
