
import logging

from fastapi import HTTPException

from src.schemas.chunking_schemas import (
    ChunkingResponse,
    CharacterChunkRequest,
    LlmChunkRequest,
    SemanticChunkRequest,
)
from src.services.chunking import chunking_service
from src.services.chunking.llm_chunker import LlmChunkingError
from src.services.chunking.semantic_chunker import SemanticChunkingError

logger = logging.getLogger(__name__)


async def chunk_by_character(data: CharacterChunkRequest) -> ChunkingResponse:
    result = await chunking_service.run_character_chunking(data)
    return ChunkingResponse(result=result)


async def chunk_by_semantic(data: SemanticChunkRequest) -> ChunkingResponse:
    try:
        result = await chunking_service.run_semantic_chunking(data)
    except SemanticChunkingError as error:
        # Log only our own safe, static error message -- never the
        # document content, and never the chained provider exception
        # (which could in principle echo back provider-side detail).
        logger.error("Semantic chunking failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate embeddings for the document",
        )

    return ChunkingResponse(result=result)


async def chunk_by_llm(data: LlmChunkRequest) -> ChunkingResponse:
    try:
        result = await chunking_service.run_llm_chunking(data)
    except LlmChunkingError as error:
        logger.error("LLM chunking failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="The AI model returned an invalid or unusable response",
        )

    return ChunkingResponse(result=result)
