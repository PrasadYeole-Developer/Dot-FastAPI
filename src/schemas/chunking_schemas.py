
from pydantic import BaseModel, Field, field_validator, model_validator

# Caps request body size so one request can't fan out into an unbounded
# number of sentences/embedding calls or an oversized LLM prompt.
MAX_DOCUMENT_LENGTH = 100_000

# NOTE: none of these have been verified against the live Gemini API (no
# live calls are made during implementation/tests, per project policy).
# - "gemini-1.5-flash" / "gemini-1.5-pro": the only two models the frontend
#   actually offers (src/components/chunking/LlmChunkConfigPanel.tsx in the
#   UI repo); "gemini-1.5-flash" is also its default.
# - "gemini-3-flash-preview": the model ai_service.py already uses for chat,
#   kept available here without changing that existing configuration.
ALLOWED_LLM_MODELS = {
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-3-flash-preview",
}


class CharacterChunkRequest(BaseModel):
    document: str = Field(max_length=MAX_DOCUMENT_LENGTH)
    chunkSize: int = Field(gt=0)
    chunkOverlap: int = Field(ge=0)

    @field_validator("document")
    @classmethod
    def document_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document must not be empty")
        return value

    @model_validator(mode="after")
    def overlap_smaller_than_chunk_size(self) -> "CharacterChunkRequest":
        if self.chunkOverlap >= self.chunkSize:
            raise ValueError("chunkOverlap must be smaller than chunkSize")
        return self


class SemanticChunkRequest(BaseModel):
    document: str = Field(max_length=MAX_DOCUMENT_LENGTH)
    similarityThreshold: float = Field(ge=0, le=1)
    minChunkSize: int = Field(gt=0)
    maxChunkSize: int = Field(gt=0)

    @field_validator("document")
    @classmethod
    def document_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document must not be empty")
        return value

    @model_validator(mode="after")
    def min_smaller_than_max(self) -> "SemanticChunkRequest":
        if self.minChunkSize >= self.maxChunkSize:
            raise ValueError("minChunkSize must be smaller than maxChunkSize")
        return self


class LlmChunkRequest(BaseModel):
    document: str = Field(max_length=MAX_DOCUMENT_LENGTH)
    model: str
    instructions: str
    maxChunkSize: int = Field(gt=0)

    @field_validator("document")
    @classmethod
    def document_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document must not be empty")
        return value

    @field_validator("model")
    @classmethod
    def model_allowed(cls, value: str) -> str:
        if value not in ALLOWED_LLM_MODELS:
            raise ValueError(
                f"model must be one of: {', '.join(sorted(ALLOWED_LLM_MODELS))}"
            )
        return value

    @field_validator("instructions")
    @classmethod
    def instructions_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be empty")
        return value


class Chunk(BaseModel):
    index: int
    content: str
    characterCount: int


class ChunkingResult(BaseModel):
    strategy: str
    chunks: list[Chunk]
    totalChunks: int
    processingTimeMs: int


class ChunkingResponse(BaseModel):
    result: ChunkingResult
