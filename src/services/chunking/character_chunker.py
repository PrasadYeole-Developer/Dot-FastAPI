
def split(document: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split document into fixed-size, overlapping character windows.

    Caller is expected to have already validated chunk_size > 0 and
    0 <= chunk_overlap < chunk_size (see chunking_schemas.py).
    """
    if not document:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    length = len(document)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(document[start:end])

        if end >= length:
            break

        start += step

    return chunks


def split_if_oversized(text: str, max_chunk_size: int) -> list[str]:
    """Return [text] unchanged if it already fits within max_chunk_size,
    otherwise deterministically re-split it into non-overlapping windows.

    Used by the semantic and LLM chunkers to guarantee max_chunk_size is
    enforced even for a single sentence/group that is itself oversized.
    """
    if len(text) <= max_chunk_size:
        return [text]

    return split(text, max_chunk_size, 0)
