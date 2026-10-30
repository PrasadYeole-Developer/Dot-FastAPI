
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ChatModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    user: str
    title: str
    lastActivity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
