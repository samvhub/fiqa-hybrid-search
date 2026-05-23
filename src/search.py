import argparse
import json
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

DATA_DIR = os.path.join(ROOT, os.environ.get("DATA_DIR", "data/fiqa"))
DENSE_MODEL = os.environ.get("DENSE_MODEL", "all-MiniLM-L6-v2")
DEFAULT_ALPHA = float(os.environ.get("DEFAULT_ALPHA", "0.5"))


def _load_fiqa_index():
    bm25_path = os.path.join(DATA_DIR, "bm25_index.pkl")
    if not os.path.exists(bm25_path):
        return None, None, None

    with open(os.path.join(DATA_DIR, "doc_ids.json"), encoding="utf-8") as f:
        doc_ids = json.load(f)
    with open(bm25_path, "rb") as f:
        bm = pickle.load(f)

    dn = DenseRetriever(model_name=DENSE_MODEL, use_model=True)
    dn.load_index(DATA_DIR, bm.docs)
    return bm, dn, doc_ids


def _load_sample_index():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "index", "sample_passages.json")
    with open(sample, "r", encoding="utf-8") as f:
        passages = json.load(f)
    docs = [p["text"] for p in passages]
    doc_ids = list(range(len(docs)))

    bm = BM25Retriever()
    bm.build_index(docs)
    dn = DenseRetriever(use_model=False)
    dn.build_index(docs)
    return bm, dn, doc_ids


def main():
    parser = argparse.ArgumentParser(description="FiQA hybrid search CLI")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    bm, dn, doc_ids = _load_fiqa_index()
    if bm is None:
        print(
            "Warning: FiQA index not found — run `make index` to build it. "
            "Falling back to 3 sample passages.",
            file=sys.stderr,
        )
        bm, dn, doc_ids = _load_sample_index()

    hy = HybridRetriever(bm, dn, alpha=DEFAULT_ALPHA)
    results = hy.search(args.query, top_k=args.top_k)

    for rank, (idx, score) in enumerate(results, start=1):
        doc_id = doc_ids[idx]
        snippet = bm.docs[idx][:200]
        print(f"{rank:>2}. score={score:.4f}  id={doc_id}  text={snippet!r}")


if __name__ == "__main__":
    main()
