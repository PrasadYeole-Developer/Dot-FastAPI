
import pytest
from pydantic import ValidationError

from src.schemas.chunking_schemas import (
    ALLOWED_LLM_MODELS,
    MAX_DOCUMENT_LENGTH,
    CharacterChunkRequest,
    LlmChunkRequest,
    SemanticChunkRequest,
)


def test_character_request_rejects_empty_document():
    with pytest.raises(ValidationError):
        CharacterChunkRequest(document="   ", chunkSize=10, chunkOverlap=0)


def test_character_request_rejects_non_positive_chunk_size():
    with pytest.raises(ValidationError):
        CharacterChunkRequest(document="hello", chunkSize=0, chunkOverlap=0)


def test_character_request_rejects_negative_overlap():
    with pytest.raises(ValidationError):
        CharacterChunkRequest(document="hello", chunkSize=10, chunkOverlap=-1)


def test_character_request_rejects_overlap_greater_or_equal_to_chunk_size():
    with pytest.raises(ValidationError):
        CharacterChunkRequest(document="hello", chunkSize=10, chunkOverlap=10)


def test_character_request_accepts_valid_payload():
    request = CharacterChunkRequest(document="hello", chunkSize=10, chunkOverlap=2)
    assert request.chunkSize == 10


def test_semantic_request_rejects_threshold_out_of_range():
    with pytest.raises(ValidationError):
        SemanticChunkRequest(
            document="hello",
            similarityThreshold=1.5,
            minChunkSize=10,
            maxChunkSize=100,
        )


def test_semantic_request_rejects_min_greater_or_equal_to_max():
    with pytest.raises(ValidationError):
        SemanticChunkRequest(
            document="hello",
            similarityThreshold=0.5,
            minChunkSize=100,
            maxChunkSize=100,
        )


def test_llm_request_rejects_unknown_model():
    with pytest.raises(ValidationError):
        LlmChunkRequest(
            document="hello",
            model="not-a-real-model",
            instructions="split it",
            maxChunkSize=500,
        )


def test_llm_request_rejects_empty_instructions():
    with pytest.raises(ValidationError):
        LlmChunkRequest(
            document="hello",
            model="gemini-1.5-flash",
            instructions="   ",
            maxChunkSize=500,
        )


def test_llm_request_accepts_valid_payload():
    request = LlmChunkRequest(
        document="hello",
        model="gemini-1.5-flash",
        instructions="split it",
        maxChunkSize=500,
    )
    assert request.model == "gemini-1.5-flash"


def test_allowed_llm_models_matches_frontend_and_existing_chatbot_model():
    # Pins the whitelist intentionally: the two models the frontend's
    # LlmChunkConfigPanel.tsx actually offers, plus the model ai_service.py
    # already uses for chat (kept available without changing that config).
    assert ALLOWED_LLM_MODELS == {
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-3-flash-preview",
    }


def test_character_request_rejects_document_over_max_length():
    too_long = "a" * (MAX_DOCUMENT_LENGTH + 1)
    with pytest.raises(ValidationError):
        CharacterChunkRequest(document=too_long, chunkSize=10, chunkOverlap=0)


def test_character_request_accepts_document_at_max_length():
    exactly_max = "a" * MAX_DOCUMENT_LENGTH
    request = CharacterChunkRequest(document=exactly_max, chunkSize=10, chunkOverlap=0)
    assert len(request.document) == MAX_DOCUMENT_LENGTH


def test_semantic_request_rejects_document_over_max_length():
    too_long = "a" * (MAX_DOCUMENT_LENGTH + 1)
    with pytest.raises(ValidationError):
        SemanticChunkRequest(
            document=too_long,
            similarityThreshold=0.5,
            minChunkSize=10,
            maxChunkSize=100,
        )


def test_llm_request_rejects_document_over_max_length():
    too_long = "a" * (MAX_DOCUMENT_LENGTH + 1)
    with pytest.raises(ValidationError):
        LlmChunkRequest(
            document=too_long,
            model="gemini-1.5-flash",
            instructions="split it",
            maxChunkSize=500,
        )
