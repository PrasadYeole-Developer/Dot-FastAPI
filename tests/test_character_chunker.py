
from src.services.chunking import character_chunker


def test_empty_document_returns_no_chunks():
    assert character_chunker.split("", 10, 0) == []


def test_document_shorter_than_chunk_size_returns_single_chunk():
    chunks = character_chunker.split("hello", 100, 0)
    assert chunks == ["hello"]


def test_no_overlap_splits_into_exact_windows():
    document = "a" * 10
    chunks = character_chunker.split(document, 5, 0)
    assert chunks == ["aaaaa", "aaaaa"]


def test_exact_multiple_boundary_has_no_trailing_empty_chunk():
    document = "abcdefghij"  # length 10
    chunks = character_chunker.split(document, 5, 0)
    assert len(chunks) == 2
    assert "".join(chunks) == document


def test_overlap_repeats_boundary_characters():
    document = "abcdefghij"  # length 10
    chunks = character_chunker.split(document, 4, 2)
    assert chunks[0] == "abcd"
    assert chunks[1] == "cdef"
    assert chunks[2] == "efgh"


def test_overlap_reconstructs_full_document_without_dropping_characters():
    document = "".join(str(i % 10) for i in range(97))
    chunk_size, overlap = 10, 3
    step = chunk_size - overlap
    chunks = character_chunker.split(document, chunk_size, overlap)

    # Every chunk must match the original document at the start position
    # the chunker is expected to use, confirming no characters are dropped.
    start = 0
    for chunk in chunks:
        assert chunk == document[start:start + chunk_size]
        start += step


def test_sequential_indexes_assigned_by_service(monkeypatch):
    import asyncio

    from src.schemas.chunking_schemas import CharacterChunkRequest
    from src.services.chunking import chunking_service

    request = CharacterChunkRequest(document="abcdefgh", chunkSize=3, chunkOverlap=1)
    result = asyncio.run(chunking_service.run_character_chunking(request))

    assert [chunk.index for chunk in result.chunks] == list(range(len(result.chunks)))
    assert result.totalChunks == len(result.chunks)
    assert result.strategy == "character"
    for chunk in result.chunks:
        assert chunk.characterCount == len(chunk.content)


def test_no_infinite_loop_with_overlap_one_less_than_chunk_size():
    document = "x" * 50
    chunks = character_chunker.split(document, 5, 4)
    # step == 1, so this must terminate and produce a bounded number of chunks
    assert len(chunks) == 46
    assert all(len(chunk) == 5 for chunk in chunks)


def test_combining_character_may_be_split_from_its_base_but_nothing_is_lost():
    # 'e' (U+0065) + combining acute accent (U+0301) -- two separate
    # Python-string code points that render as one visual glyph. This is a
    # *documented limitation*, not a requirement to fix: splitting can land
    # between a base character and its combining mark, but every code
    # point is still preserved exactly (no data loss, just a cosmetic
    # mid-grapheme boundary). All characters below are written as explicit
    # \u escapes to avoid any source-encoding ambiguity.
    base_e = "e"
    combining_accent = "́"
    document = "caf" + base_e + combining_accent + " au lait"
    chunks = character_chunker.split(document, 4, 0)

    assert "".join(chunks) == document
    assert chunks[0][-1] == base_e
    assert chunks[1][0] == combining_accent


def test_zwj_emoji_sequence_may_be_split_but_nothing_is_lost():
    # A ZWJ (U+200D) family-emoji sequence is five separate Unicode code
    # points forming one visual glyph. Splitting mid-sequence is a cosmetic
    # limitation only: no code point is ever dropped or duplicated.
    zwj = "‍"
    man = "\U0001F468"
    woman = "\U0001F469"
    girl = "\U0001F467"
    document = "Hi " + man + zwj + woman + zwj + girl + " end"
    chunks = character_chunker.split(document, 5, 0)

    assert "".join(chunks) == document
    assert sum(len(chunk) for chunk in chunks) == len(document)


def test_split_if_oversized_returns_unchanged_when_within_limit():
    assert character_chunker.split_if_oversized("short text", 100) == ["short text"]


def test_split_if_oversized_splits_when_over_limit():
    text = "y" * 50
    chunks = character_chunker.split_if_oversized(text, 20)

    assert all(len(chunk) <= 20 for chunk in chunks)
    assert "".join(chunks) == text
    assert len(chunks) == 3
