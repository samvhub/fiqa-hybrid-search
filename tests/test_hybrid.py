from src.retrieval.hybrid import HybridRetriever

def test_normalize_and_combine():
    class Dummy:
        def __init__(self, pairs):
            self._pairs = pairs
        def search(self, q, top_k=10):
            return self._pairs

    bm = Dummy([(0, 1.0), (1, 0.5)])
    dn = Dummy([(1, 2.0), (2, 0.1)])
    h = HybridRetriever(bm, dn, alpha=0.6)
    res = h.search("q", top_k=3)
    assert isinstance(res, list)
    assert all(len(x) == 2 for x in res)
