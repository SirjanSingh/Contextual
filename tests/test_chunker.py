from repo_aware_ai.chunker import chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "abcdefghij" * 100  # 1000 chars
    chunks = chunk_text(text, source="x.py", chunk_size=400, overlap=50)

    assert len(chunks) >= 2
    # Each chunk's text matches the slice it claims.
    for c in chunks:
        assert text[c.start_char : c.end_char] == c.text
        assert c.end_char - c.start_char <= 400

    # Overlap: subsequent chunk starts inside the previous chunk's range.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_char >= prev.start_char
        assert nxt.start_char < prev.end_char


def test_chunk_text_short_input_yields_one_chunk():
    chunks = chunk_text("hello", source="a.txt", chunk_size=400, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 5


def test_chunk_text_skips_whitespace_only_blocks():
    chunks = chunk_text("    \n\n   ", source="a.txt", chunk_size=400, overlap=50)
    assert chunks == []


def test_chunk_text_overlap_validation():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", source="a.py", chunk_size=10, overlap=10)
