import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

def load_passages():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "index", "sample_passages.json")
    with open(sample, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    passages = load_passages()
    docs = [p["text"] for p in passages]

    bm = BM25Retriever()
    bm.build_index(docs)

    dn = DenseRetriever()
    dn.build_index(docs)

    hy = HybridRetriever(bm, dn)

    results = hy.search(args.query, top_k=args.top_k)
    for rank, (idx, score) in enumerate(results, start=1):
        print(f"{rank}. score={score:.4f} id={idx} text={docs[idx][:200]!r}")

if __name__ == '__main__':
    main()
