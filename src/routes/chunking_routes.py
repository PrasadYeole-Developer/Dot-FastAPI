
from fastapi import APIRouter, Depends

from src.controllers.chunking_controller import (
    chunk_by_character,
    chunk_by_llm,
    chunk_by_semantic,
)
from src.middlewares.auth_middleware import auth_user
from src.schemas.chunking_schemas import (
    ChunkingResponse,
    CharacterChunkRequest,
    LlmChunkRequest,
    SemanticChunkRequest,
)

router = APIRouter()


@router.post("/character", response_model=ChunkingResponse)
async def chunk_by_character_route(
    data: CharacterChunkRequest,
    user: dict = Depends(auth_user),
):
    return await chunk_by_character(data)


@router.post("/semantic", response_model=ChunkingResponse)
async def chunk_by_semantic_route(
    data: SemanticChunkRequest,
    user: dict = Depends(auth_user),
):
    return await chunk_by_semantic(data)


@router.post("/llm", response_model=ChunkingResponse)
async def chunk_by_llm_route(
    data: LlmChunkRequest,
    user: dict = Depends(auth_user),
):
    return await chunk_by_llm(data)
