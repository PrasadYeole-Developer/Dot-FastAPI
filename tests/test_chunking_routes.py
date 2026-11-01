
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.middlewares.auth_middleware import auth_user
from src.services.chunking import llm_chunker, semantic_chunker

FAKE_USER = {"_id": "000000000000000000000001", "email": "test@example.com"}


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[auth_user] = lambda: FAKE_USER
    yield
    app.dependency_overrides.pop(auth_user, None)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_character_route_returns_expected_shape(client):
    response = await client.post(
        "/api/chunking/character",
        json={"document": "abcdefghij", "chunkSize": 5, "chunkOverlap": 0},
    )

    assert response.status_code == 200
    body = response.json()
    result = body["result"]

    assert result["strategy"] == "character"
    assert result["totalChunks"] == 2
    assert result["chunks"][0] == {"index": 0, "content": "abcde", "characterCount": 5}
    assert result["chunks"][1] == {"index": 1, "content": "fghij", "characterCount": 5}
    assert isinstance(result["processingTimeMs"], int)


async def test_character_route_rejects_invalid_chunk_size(client):
    response = await client.post(
        "/api/chunking/character",
        json={"document": "abcdefghij", "chunkSize": 0, "chunkOverlap": 0},
    )

    assert response.status_code == 422


async def test_character_route_rejects_empty_document(client):
    response = await client.post(
        "/api/chunking/character",
        json={"document": "   ", "chunkSize": 5, "chunkOverlap": 0},
    )

    assert response.status_code == 422


async def test_character_route_requires_authentication():
    app.dependency_overrides.pop(auth_user, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chunking/character",
            json={"document": "abc", "chunkSize": 5, "chunkOverlap": 0},
        )

    assert response.status_code == 401


async def test_semantic_route_returns_expected_shape(client, monkeypatch):
    mock = AsyncMock(return_value=[1.0, 0.0])
    monkeypatch.setattr(semantic_chunker, "generate_vectors", mock)

    response = await client.post(
        "/api/chunking/semantic",
        json={
            "document": "First sentence here. Second sentence here.",
            "similarityThreshold": 0.5,
            "minChunkSize": 1,
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["strategy"] == "semantic"
    assert result["totalChunks"] >= 1


async def test_semantic_route_rejects_invalid_threshold(client):
    response = await client.post(
        "/api/chunking/semantic",
        json={
            "document": "Some text.",
            "similarityThreshold": 2.0,
            "minChunkSize": 1,
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 422


async def test_semantic_route_returns_502_on_provider_failure(client, monkeypatch):
    mock = AsyncMock(side_effect=RuntimeError("provider down"))
    monkeypatch.setattr(semantic_chunker, "generate_vectors", mock)

    response = await client.post(
        "/api/chunking/semantic",
        json={
            "document": "First sentence here. Second sentence here.",
            "similarityThreshold": 0.5,
            "minChunkSize": 1,
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 502
    assert "provider down" not in response.text


async def test_llm_route_returns_expected_shape(client, monkeypatch):
    document = "First sentence here. Second sentence here."
    payload = {"groups": [[0], [1]]}
    mock = AsyncMock(return_value=SimpleNamespace(text=json.dumps(payload)))
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock))
    )
    monkeypatch.setattr(llm_chunker, "ai", fake_client)

    response = await client.post(
        "/api/chunking/llm",
        json={
            "document": document,
            "model": "gemini-1.5-flash",
            "instructions": "split semantically",
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["strategy"] == "llm"
    # Content comes from the backend-maintained original sentences, not
    # anything the model echoed back -- confirmed exactly here.
    assert result["chunks"][0]["content"] == "First sentence here."
    assert result["chunks"][1]["content"] == "Second sentence here."


async def test_llm_route_rejects_unknown_model(client):
    response = await client.post(
        "/api/chunking/llm",
        json={
            "document": "First sentence here.",
            "model": "made-up-model",
            "instructions": "split semantically",
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 422


async def test_llm_route_returns_502_on_malformed_model_output(client, monkeypatch):
    mock = AsyncMock(return_value=SimpleNamespace(text="not json"))
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock))
    )
    monkeypatch.setattr(llm_chunker, "ai", fake_client)

    response = await client.post(
        "/api/chunking/llm",
        json={
            # Needs >= 2 sentences: a single-sentence document skips the
            # model entirely via the shortcut, which would never exercise
            # this failure path.
            "document": "First sentence here. Second sentence here.",
            "model": "gemini-1.5-flash",
            "instructions": "split semantically",
            "maxChunkSize": 1000,
        },
    )

    assert response.status_code == 502


async def test_llm_route_logs_provider_failure_without_leaking_details(client, monkeypatch, caplog):
    mock = AsyncMock(side_effect=RuntimeError("API key AIzaSecretValue123 rejected"))
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock))
    )
    monkeypatch.setattr(llm_chunker, "ai", fake_client)

    with caplog.at_level(logging.ERROR):
        response = await client.post(
            "/api/chunking/llm",
            json={
                "document": "First sentence here. Second sentence here.",
                "model": "gemini-1.5-flash",
                "instructions": "split semantically",
                "maxChunkSize": 1000,
            },
        )

    assert response.status_code == 502
    assert "AIzaSecretValue123" not in response.text
    assert any("LLM chunking failed" in record.message for record in caplog.records)
    assert not any("AIzaSecretValue123" in record.message for record in caplog.records)


async def test_semantic_route_requires_authentication():
    app.dependency_overrides.pop(auth_user, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chunking/semantic",
            json={
                "document": "Some text.",
                "similarityThreshold": 0.5,
                "minChunkSize": 1,
                "maxChunkSize": 100,
            },
        )

    assert response.status_code == 401


async def test_llm_route_requires_authentication():
    app.dependency_overrides.pop(auth_user, None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/chunking/llm",
            json={
                "document": "Some text.",
                "model": "gemini-1.5-flash",
                "instructions": "split it",
                "maxChunkSize": 100,
            },
        )

    assert response.status_code == 401


async def test_character_route_rejects_document_over_max_length(client):
    from src.schemas.chunking_schemas import MAX_DOCUMENT_LENGTH

    response = await client.post(
        "/api/chunking/character",
        json={
            "document": "a" * (MAX_DOCUMENT_LENGTH + 1),
            "chunkSize": 5,
            "chunkOverlap": 0,
        },
    )

    assert response.status_code == 422
