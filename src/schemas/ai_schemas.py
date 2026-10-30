from pydantic import BaseModel, Field


class AIResponse(BaseModel):
    intent: str = Field(
        description="The user's intent, such as greeting, question, or request."
    )

    message: str = Field(
        description="The actual response to display to the user."
    )

    sentiment: str = Field(
        description="The sentiment of the user's message."
    )

    needs_human_support: bool = Field(
        description="Whether the user needs assistance from a human."
    )