from rank_bm25 import BM25Okapi
from typing import List

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.docs = []

    def build_index(self, docs: List[str]):
        tokenized = [d.split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs

    def search(self, query: str, top_k: int = 10):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
