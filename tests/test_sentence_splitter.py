
from src.services.chunking import sentence_splitter


def test_empty_document_returns_no_sentences():
    sentences, separators = sentence_splitter.split_with_separators("")
    assert sentences == []
    assert separators == []


def test_whitespace_only_document_returns_no_sentences():
    sentences, separators = sentence_splitter.split_with_separators("   \n\t  ")
    assert sentences == []
    assert separators == []


def test_single_sentence_no_separator():
    sentences, separators = sentence_splitter.split_with_separators("Just one sentence.")
    assert sentences == ["Just one sentence."]
    assert separators == [""]


def test_multiple_sentences_split_correctly():
    document = "First one. Second one. Third one."
    sentences, separators = sentence_splitter.split_with_separators(document)
    assert sentences == ["First one.", "Second one.", "Third one."]
    assert separators == [" ", " ", ""]


def test_paragraph_break_preserved_as_separator():
    document = "First paragraph sentence.\n\nSecond paragraph sentence."
    sentences, separators = sentence_splitter.split_with_separators(document)
    assert sentences == ["First paragraph sentence.", "Second paragraph sentence."]
    assert separators == ["\n\n", ""]


def test_join_span_reconstructs_full_document_losslessly():
    document = "First one.\n\nSecond one.   Third one.\tFourth one."
    sentences, separators = sentence_splitter.split_with_separators(document)
    reconstructed = sentence_splitter.join_span(sentences, separators, 0, len(sentences) - 1)
    assert reconstructed == document.strip()


def test_join_span_subset_excludes_boundary_separator():
    document = "First one. Second one. Third one."
    sentences, separators = sentence_splitter.split_with_separators(document)
    first_two = sentence_splitter.join_span(sentences, separators, 0, 1)
    assert first_two == "First one. Second one."


def test_document_with_no_terminal_punctuation_is_one_sentence():
    document = "no punctuation at all here"
    sentences, separators = sentence_splitter.split_with_separators(document)
    assert sentences == [document]
    assert separators == [""]


def test_no_sentence_segment_is_ever_blank():
    document = "Wait...  Really?   Yes. Okay then."
    sentences, separators = sentence_splitter.split_with_separators(document)
    assert all(s.strip() != "" for s in sentences)
    assert sentence_splitter.join_span(sentences, separators, 0, len(sentences) - 1) == document.strip()
