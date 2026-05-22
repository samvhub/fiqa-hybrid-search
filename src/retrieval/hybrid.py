from typing import List, Tuple
import numpy as np

class HybridRetriever:
    def __init__(self, bm25, dense, alpha: float = 0.5):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha

    def _normalize_scores(self, pairs):
        # pairs: list of (idx, score)
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        mn, mx = min(vals), max(vals)
        denom = mx - mn if mx != mn else 1.0
        return {i: (s - mn) / denom for i, s in pairs}

    def search(self, query: str, top_k: int = 10):
        bm = self.bm25.search(query, top_k=top_k*5)
        dn = self.dense.search(query, top_k=top_k*5)

        bm_map = self._normalize_scores(bm)
        dn_map = self._normalize_scores(dn)

        # union of candidate ids
        ids = set(list(bm_map.keys()) + list(dn_map.keys()))
        combined = []
        for i in ids:
            s_b = bm_map.get(i, 0.0)
            s_d = dn_map.get(i, 0.0)
            score = self.alpha * s_d + (1 - self.alpha) * s_b
            combined.append((i, score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
