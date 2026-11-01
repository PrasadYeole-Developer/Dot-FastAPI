
import time

from src.schemas.chunking_schemas import (
    Chunk,
    ChunkingResult,
    CharacterChunkRequest,
    LlmChunkRequest,
    SemanticChunkRequest,
)
from src.services.chunking import character_chunker, llm_chunker, semantic_chunker


def _build_result(
    strategy: str,
    chunk_texts: list[str],
    started_at: float,
) -> ChunkingResult:
    chunks = [
        Chunk(index=index, content=text, characterCount=len(text))
        for index, text in enumerate(chunk_texts)
    ]

    elapsed_ms = round((time.perf_counter() - started_at) * 1000)

    return ChunkingResult(
        strategy=strategy,
        chunks=chunks,
        totalChunks=len(chunks),
        processingTimeMs=elapsed_ms,
    )


async def run_character_chunking(data: CharacterChunkRequest) -> ChunkingResult:
    started_at = time.perf_counter()

    chunk_texts = character_chunker.split(
        data.document,
        data.chunkSize,
        data.chunkOverlap,
    )

    return _build_result("character", chunk_texts, started_at)


async def run_semantic_chunking(data: SemanticChunkRequest) -> ChunkingResult:
    started_at = time.perf_counter()

    chunk_texts = await semantic_chunker.split(
        data.document,
        data.similarityThreshold,
        data.minChunkSize,
        data.maxChunkSize,
    )

    return _build_result("semantic", chunk_texts, started_at)


async def run_llm_chunking(data: LlmChunkRequest) -> ChunkingResult:
    started_at = time.perf_counter()

    chunk_texts = await llm_chunker.split(
        data.document,
        data.model,
        data.instructions,
        data.maxChunkSize,
    )

    return _build_result("llm", chunk_texts, started_at)
