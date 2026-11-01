
import asyncio

from src.services.ai_service import generate_vectors
from src.services.chunking import character_chunker, sentence_splitter

# Caps how many Gemini embedding requests this module will have in flight
# at once for a single chunking request, regardless of document size.
_MAX_CONCURRENT_EMBEDDINGS = 8


class SemanticChunkingError(Exception):
    """Raised when embeddings cannot be generated or are unusable."""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    # Dimensions are guaranteed equal by _generate_embeddings' validation
    # before this is ever called, so zip here never silently truncates.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _incremental_mean(mean: list[float], count: int, new_value: list[float]) -> list[float]:
    """Mathematically correct running mean: each new embedding contributes
    weight 1/(count+1), unlike a naive repeated pairwise average."""
    return [(m * count + v) / (count + 1) for m, v in zip(mean, new_value)]


async def _fetch_embedding(sentence: str, semaphore: asyncio.Semaphore) -> list[float]:
    async with semaphore:
        try:
            vector = await generate_vectors(sentence)
        except Exception as error:
            raise SemanticChunkingError(
                "Failed to generate an embedding for the document"
            ) from error

    if not vector:
        raise SemanticChunkingError("Embedding provider returned an empty vector")

    return vector


async def _generate_embeddings(sentences: list[str]) -> list[list[float]]:
    # Dedupe identical sentence text so repeated content isn't re-embedded.
    unique_sentences = list(dict.fromkeys(sentences))
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_EMBEDDINGS)

    tasks = [
        asyncio.ensure_future(_fetch_embedding(sentence, semaphore))
        for sentence in unique_sentences
    ]

    try:
        unique_results = await asyncio.gather(*tasks)
    except Exception:
        # asyncio.gather does not cancel sibling tasks on failure by
        # default -- without this, a failed embedding call would leave
        # every other in-flight Gemini request running unattended.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    expected_dim = len(unique_results[0])
    for vector in unique_results:
        if len(vector) != expected_dim:
            raise SemanticChunkingError(
                "Embedding provider returned vectors with inconsistent dimensions"
            )

    embedding_by_sentence = dict(zip(unique_sentences, unique_results))
    return [embedding_by_sentence[sentence] for sentence in sentences]


async def split(
    document: str,
    similarity_threshold: float,
    min_chunk_size: int,
    max_chunk_size: int,
) -> list[str]:
    sentences, separators = sentence_splitter.split_with_separators(document)

    if not sentences:
        return []

    if len(sentences) == 1:
        return character_chunker.split_if_oversized(sentences[0], max_chunk_size)

    whole_text = sentence_splitter.join_span(sentences, separators, 0, len(sentences) - 1)
    if len(whole_text) <= min_chunk_size:
        return character_chunker.split_if_oversized(whole_text, max_chunk_size)

    embeddings = await _generate_embeddings(sentences)

    chunks: list[str] = []
    chunk_start = 0
    running_mean = embeddings[0]
    running_count = 1

    for i in range(1, len(sentences)):
        current_text = sentence_splitter.join_span(sentences, separators, chunk_start, i - 1)
        candidate_text = current_text + separators[i - 1] + sentences[i]
        similarity = _cosine_similarity(running_mean, embeddings[i])

        fits_within_max = len(candidate_text) <= max_chunk_size
        below_min = len(current_text) < min_chunk_size

        if fits_within_max and (similarity >= similarity_threshold or below_min):
            running_mean = _incremental_mean(running_mean, running_count, embeddings[i])
            running_count += 1
            continue

        chunks.extend(character_chunker.split_if_oversized(current_text, max_chunk_size))
        chunk_start = i
        running_mean = embeddings[i]
        running_count = 1

    final_text = sentence_splitter.join_span(sentences, separators, chunk_start, len(sentences) - 1)
    chunks.extend(character_chunker.split_if_oversized(final_text, max_chunk_size))

    return chunks
