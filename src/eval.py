import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

def run_quick_bench():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "index", "sample_passages.json")
    with open(sample, "r", encoding="utf-8") as f:
        passages = json.load(f)
    docs = [p["text"] for p in passages]

    bm = BM25Retriever(); bm.build_index(docs)
    dn = DenseRetriever(); dn.build_index(docs)
    hy = HybridRetriever(bm, dn)

    queries = ["short selling", "market order", "diversification portfolio risk"]
    out = {}
    for name, retr in [("bm25", bm), ("dense", dn), ("hybrid", hy)]:
        times = []
        for q in queries:
            t0 = time.time()
            retr.search(q, top_k=5)
            times.append((time.time() - t0) * 1000.0)
        out[name] = {"p50_ms": sorted(times)[len(times)//2], "p95_ms": sorted(times)[-1]}

    os.makedirs(os.path.join(os.path.dirname(here), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(here), "results", "bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote results/bench.json")

if __name__ == '__main__':
    run_quick_bench()
