def chunk_text(text: str, max_tokens: int = 100):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])
