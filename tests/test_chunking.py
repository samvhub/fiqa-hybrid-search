from src.retrieval.utils import chunk_text

def test_chunk_text_small():
    text = "one two three four five"
    chunks = list(chunk_text(text, max_tokens=2))
    assert chunks == ["one two", "three four", "five"]
