
import re

# Captures the whitespace immediately after a sentence-ending punctuation
# mark as its own group, so splitting never discards it. Every match
# requires a preceding [.!?], so a split point can never produce a blank
# sentence segment (the punctuation always belongs to the segment before it).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(\s+)")


def split_with_separators(document: str) -> tuple[list[str], list[str]]:
    """Split document into sentences while preserving the exact original
    whitespace between them (paragraph breaks, double spaces, tabs, ...).

    Returns (sentences, separators), where separators[i] is the exact
    whitespace that followed sentences[i] in the original (outer-stripped)
    document. separators[-1] is "" since nothing follows the last sentence.

    join_span(sentences, separators, 0, len(sentences) - 1) always
    reconstructs document.strip() exactly -- no characters are lost.
    """
    stripped = document.strip()

    if not stripped:
        return [], []

    parts = _SENTENCE_SPLIT_RE.split(stripped)
    sentences = parts[0::2]
    separators = parts[1::2]

    while len(separators) < len(sentences):
        separators.append("")

    return sentences, separators


def join_span(
    sentences: list[str],
    separators: list[str],
    start: int,
    end: int,
) -> str:
    """Reconstruct the original text spanning sentences[start:end+1],
    including the original separators between them (but not before start
    or after end -- a span boundary is a deliberate cut point)."""
    pieces = []

    for index in range(start, end + 1):
        pieces.append(sentences[index])
        if index != end:
            pieces.append(separators[index])

    return "".join(pieces)
