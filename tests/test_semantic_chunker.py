
import asyncio
import math
from unittest.mock import AsyncMock

import pytest

from src.services.chunking import semantic_chunker
from src.services.chunking.semantic_chunker import SemanticChunkingError

PARALLEL = [1.0, 0.0]
ORTHOGONAL_A = [1.0, 0.0]
ORTHOGONAL_B = [0.0, 1.0]


def _patch_vectors(monkeypatch, mapping: dict[str, list[float]]):
    mock = AsyncMock(side_effect=lambda sentence: mapping[sentence])
    monkeypatch.setattr(semantic_chunker, "generate_vectors", mock)
    return mock


async def test_empty_document_returns_no_chunks():
    chunks = await semantic_chunker.split("", 0.75, 100, 800)
    assert chunks == []


async def test_single_sentence_skips_embedding_calls(monkeypatch):
    mock = _patch_vectors(monkeypatch, {})
    chunks = await semantic_chunker.split("Just one sentence.", 0.75, 100, 800)

    assert chunks == ["Just one sentence."]
    mock.assert_not_called()


async def test_short_document_under_min_size_skips_embedding_calls(monkeypatch):
    mock = _patch_vectors(monkeypatch, {})
    document = "Hi there. Short doc."
    chunks = await semantic_chunker.split(document, 0.75, 1000, 2000)

    assert chunks == [document]
    mock.assert_not_called()


async def test_high_similarity_sentences_merge_into_one_chunk(monkeypatch):
    sentences = [
        "Cats are small mammals.",
        "Dogs are also mammals.",
        "Both are common household pets.",
    ]
    mapping = {s: PARALLEL for s in sentences}
    _patch_vectors(monkeypatch, mapping)

    document = " ".join(sentences)
    chunks = await semantic_chunker.split(document, 0.9, 1, 1000)

    assert chunks == [document]


async def test_low_similarity_sentences_split_into_separate_chunks(monkeypatch):
    sentences = [
        "Topic one is about cooking.",
        "Topic two is about rockets.",
        "Topic three is about painting.",
    ]
    vectors = [ORTHOGONAL_A, ORTHOGONAL_B, ORTHOGONAL_A]
    mapping = dict(zip(sentences, vectors))
    _patch_vectors(monkeypatch, mapping)

    document = " ".join(sentences)
    chunks = await semantic_chunker.split(document, 0.9, 1, 1000)

    assert chunks == sentences


async def test_max_chunk_size_forces_split_despite_high_similarity(monkeypatch):
    sentences = [
        "Sentence number one is here.",
        "Sentence number two is here.",
        "Sentence number three is here.",
    ]
    mapping = {s: PARALLEL for s in sentences}
    _patch_vectors(monkeypatch, mapping)

    document = " ".join(sentences)
    # Each sentence is ~29 chars; cap max size so only one sentence fits per chunk.
    chunks = await semantic_chunker.split(document, 0.5, 1, 30)

    assert chunks == sentences
    assert all(len(chunk) <= 30 for chunk in chunks)


async def test_min_chunk_size_forces_merge_despite_low_similarity(monkeypatch):
    sentences = ["Short one.", "Short two.", "Short three."]
    vectors = [ORTHOGONAL_A, ORTHOGONAL_B, ORTHOGONAL_A]
    mapping = dict(zip(sentences, vectors))
    _patch_vectors(monkeypatch, mapping)

    document = " ".join(sentences)
    # min_chunk_size forces the first two (dissimilar) sentences together.
    chunks = await semantic_chunker.split(document, 0.9, 15, 1000)

    assert len(chunks) == 2
    assert chunks[0] == "Short one. Short two."
    assert chunks[1] == "Short three."


async def test_chunk_ordering_matches_document_order(monkeypatch):
    sentences = ["Alpha sentence.", "Beta sentence.", "Gamma sentence."]
    vectors = [ORTHOGONAL_A, ORTHOGONAL_B, ORTHOGONAL_A]
    mapping = dict(zip(sentences, vectors))
    _patch_vectors(monkeypatch, mapping)

    document = " ".join(sentences)
    chunks = await semantic_chunker.split(document, 0.9, 1, 1000)

    assert chunks == sentences


# --- Fixes for the FAIL-verdict review findings below ---------------------


async def test_single_oversized_sentence_is_split_to_respect_max_chunk_size(monkeypatch):
    mock = _patch_vectors(monkeypatch, {})
    long_sentence = ("word " * 200).strip() + "."  # 1000 chars, one sentence

    chunks = await semantic_chunker.split(long_sentence, 0.75, 10, 100)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks) == long_sentence
    mock.assert_not_called()  # single-sentence shortcut never calls the provider


async def test_oversized_sentence_inside_multi_sentence_document_is_split(monkeypatch):
    long_sentence = ("word " * 200).strip() + "."  # 1000 chars
    short_sentence = "Short one here."
    mapping = {long_sentence: PARALLEL, short_sentence: ORTHOGONAL_B}
    _patch_vectors(monkeypatch, mapping)

    document = f"{long_sentence} {short_sentence}"
    chunks = await semantic_chunker.split(document, 0.9, 1, 100)

    assert all(len(chunk) <= 100 for chunk in chunks)
    # The long sentence's own split pieces reconstruct it exactly; the
    # short sentence (dissimilar, so not merged) survives as its own chunk.
    long_pieces = chunks[:-1]
    assert "".join(long_pieces) == long_sentence
    assert chunks[-1] == short_sentence


def test_incremental_mean_matches_true_average_over_many_vectors():
    vectors = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [4.0, 0.0], [0.0, 4.0]]

    mean = vectors[0]
    count = 1
    for vector in vectors[1:]:
        mean = semantic_chunker._incremental_mean(mean, count, vector)
        count += 1

    true_mean = [sum(v[dim] for v in vectors) / len(vectors) for dim in range(2)]

    assert mean == pytest.approx(true_mean)


def test_incremental_mean_initializes_correctly_from_first_embedding():
    first = [3.0, -1.0, 2.0]
    # Merging a second, identical vector with count=1 must reproduce the
    # ordinary two-element average -- the formula's base case.
    merged = semantic_chunker._incremental_mean(first, 1, [5.0, 1.0, 0.0])
    assert merged == pytest.approx([4.0, 0.0, 1.0])


async def test_running_mean_fix_changes_the_grouping_decision(monkeypatch):
    # Three sentences clustered together (0deg, 30deg, 60deg on the unit
    # circle) followed by a fourth sentence (90deg) that is meaningfully
    # further away. A mathematically correct running mean of the first
    # three sentences is measurably different from the old buggy
    # exponentially-weighted average -- enough to flip whether sentence
    # three merges in, for a threshold set exactly between the two.
    e0 = [1.0, 0.0]
    e1 = [math.cos(math.radians(30)), math.sin(math.radians(30))]
    e2 = [math.cos(math.radians(60)), math.sin(math.radians(60))]
    e3 = [0.0, 1.0]

    sentences = [
        "Sentence zero about topic A.",
        "Sentence one about topic A.",
        "Sentence two about topic A.",
        "Sentence three about topic B.",
    ]
    mapping = dict(zip(sentences, [e0, e1, e2, e3]))
    _patch_vectors(monkeypatch, mapping)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.hypot(*a) * math.hypot(*b))

    mean_2 = [(e0[i] + e1[i]) / 2 for i in range(2)]
    true_mean_3 = [(e0[i] + e1[i] + e2[i]) / 3 for i in range(2)]
    buggy_mean_3 = [(mean_2[i] + e2[i]) / 2 for i in range(2)]

    # Sanity-check the scenario: the first two merges must clear any
    # reasonable threshold (they're scheme-independent, n<=2), and the
    # correct vs. buggy third-sentence comparisons must actually diverge.
    threshold = (cosine(true_mean_3, e3) + cosine(buggy_mean_3, e3)) / 2
    assert cosine(e0, e1) > threshold
    assert cosine(mean_2, e2) > threshold
    assert cosine(true_mean_3, e3) < threshold < cosine(buggy_mean_3, e3)

    document = " ".join(sentences)
    chunks = await semantic_chunker.split(document, threshold, 1, 100_000)

    # Corrected running mean: sentence three falls below threshold and
    # starts its own chunk. The old buggy average would have scored it
    # above threshold and incorrectly merged all four sentences together.
    assert len(chunks) == 2
    assert chunks[1] == sentences[3]


async def test_empty_embedding_vector_is_rejected(monkeypatch):
    async def fake_generate_vectors(sentence):
        return []

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)
    sentences = ["Sentence one here.", "Sentence two here."]
    document = " ".join(sentences)

    with pytest.raises(SemanticChunkingError):
        await semantic_chunker.split(document, 0.5, 1, 100_000)


async def test_zero_magnitude_vector_similarity_is_handled_safely(monkeypatch):
    async def fake_generate_vectors(sentence):
        return [0.0, 0.0]

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)
    sentences = ["Sentence one here.", "Sentence two here.", "Sentence three here."]
    document = " ".join(sentences)

    # Zero-magnitude vectors must not raise (no division by zero); cosine
    # similarity is treated as 0 (maximally dissimilar), so each sentence
    # starts its own chunk rather than merging.
    chunks = await semantic_chunker.split(document, 0.9, 1, 100_000)
    assert chunks == sentences


async def test_embedding_dimension_mismatch_is_rejected(monkeypatch):
    async def fake_generate_vectors(sentence):
        return [1.0, 0.0, 0.0] if "two" in sentence else [1.0, 0.0]

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)
    sentences = ["Sentence one here.", "Sentence two here.", "Sentence three here."]
    document = " ".join(sentences)

    with pytest.raises(SemanticChunkingError):
        await semantic_chunker.split(document, 0.5, 1, 100_000)


async def test_paragraph_break_is_preserved_within_a_merged_chunk(monkeypatch):
    s1 = "First paragraph sentence."
    s2 = "Second paragraph sentence."
    document = f"{s1}\n\n{s2}"
    _patch_vectors(monkeypatch, {s1: PARALLEL, s2: PARALLEL})

    chunks = await semantic_chunker.split(document, 0.9, 1, 1000)

    assert chunks == [document]
    assert "\n\n" in chunks[0]


async def test_repeated_identical_sentence_is_embedded_only_once(monkeypatch):
    calls = []

    async def fake_generate_vectors(sentence):
        calls.append(sentence)
        return PARALLEL

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)

    repeated = "This sentence repeats verbatim."
    other = "A different sentence entirely."
    document = f"{repeated} {other} {repeated}"

    await semantic_chunker.split(document, 0.9, 1, 100_000)

    assert calls.count(repeated) == 1


async def test_embedding_concurrency_is_bounded(monkeypatch):
    current = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_generate_vectors(sentence):
        nonlocal current, max_concurrent
        async with lock:
            current += 1
            max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)
        async with lock:
            current -= 1
        return PARALLEL

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)

    sentences = [f"Sentence number {i} appears here." for i in range(30)]
    document = " ".join(sentences)

    await semantic_chunker.split(document, 0.5, 1, 1_000_000)

    assert max_concurrent <= semantic_chunker._MAX_CONCURRENT_EMBEDDINGS
    assert max_concurrent > 1  # confirm calls actually overlapped, not fully serial


async def test_partial_embedding_failure_cancels_pending_tasks(monkeypatch):
    cancelled = []

    async def fake_generate_vectors(sentence):
        if sentence == "Sentence two here.":
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(sentence)
            raise
        return PARALLEL

    monkeypatch.setattr(semantic_chunker, "generate_vectors", fake_generate_vectors)

    sentences = [
        "Sentence one here.",
        "Sentence two here.",
        "Sentence three here.",
        "Sentence four here.",
    ]
    document = " ".join(sentences)

    with pytest.raises(SemanticChunkingError):
        await semantic_chunker.split(document, 0.5, 1, 100_000)

    assert len(cancelled) > 0
