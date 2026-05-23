import csv
import json
import os
import pickle
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data", "fiqa")
FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def _download_fiqa(data_dir: str) -> None:
    # Skip everything if corpus is already in place (e.g. placed manually)
    if os.path.exists(os.path.join(data_dir, "corpus.jsonl")):
        return

    parent = os.path.dirname(data_dir)
    os.makedirs(parent, exist_ok=True)
    zip_path = os.path.join(parent, "fiqa.zip")
    if not os.path.exists(zip_path):
        print("Downloading FiQA dataset (~18 MB)...")
        import warnings
        import requests
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        with requests.get(FIQA_URL, stream=True, verify=False, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            received = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    received += len(chunk)
                    if total:
                        print(f"\r  {received/1e6:.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print(f"\nDownload complete ({received/1e6:.1f} MB).")
    print("Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent)
    print("Extraction complete.")


def _load_corpus(data_dir: str) -> tuple:
    doc_ids, docs = [], []
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            doc_ids.append(obj["_id"])
            docs.append(obj["text"])
    return doc_ids, docs


def _load_queries(data_dir: str) -> dict:
    queries = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            queries[obj["_id"]] = obj["text"]
    return queries


def _load_qrels(data_dir: str, split: str = "dev") -> dict:
    qrels = {}
    path = os.path.join(data_dir, "qrels", f"{split}.tsv")
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) == 3:
                qid, did, score = row[0], row[1], int(row[2])
            elif len(row) == 4:
                qid, _, did, score = row[0], row[1], row[2], int(row[3])
            else:
                continue
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels


def build_fiqa_index(data_dir: str = DATA_DIR) -> str:
    """Download FiQA from BEIR, build BM25 + dense indexes, persist to data_dir."""
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever

    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    dense_path = os.path.join(data_dir, "dense_embeddings.npy")
    if os.path.exists(bm25_path) and os.path.exists(dense_path):
        print("Pre-built index found, skipping rebuild.")
        return data_dir

    _download_fiqa(data_dir)

    print("Loading corpus...")
    doc_ids, docs = _load_corpus(data_dir)
    print(f"  {len(docs)} passages.")

    print("Loading queries and qrels...")
    queries = _load_queries(data_dir)
    qrels = _load_qrels(data_dir, split="dev")
    dev_qids = set(qrels.keys())
    dev_queries = {qid: q for qid, q in queries.items() if qid in dev_qids}
    print(f"  {len(dev_queries)} dev queries with relevance judgments.")

    with open(os.path.join(data_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    with open(os.path.join(data_dir, "dev_queries.json"), "w", encoding="utf-8") as f:
        json.dump(dev_queries, f)
    with open(os.path.join(data_dir, "dev_qrels.json"), "w", encoding="utf-8") as f:
        json.dump(qrels, f)

    print("Building BM25 index...")
    bm = BM25Retriever()
    bm.build_index(docs)
    bm25_path = os.path.join(data_dir, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm, f)
    print(f"  BM25 saved -> {bm25_path}")

    print("Encoding dense embeddings with all-MiniLM-L6-v2 (CPU, this takes a few minutes)...")
    dn = DenseRetriever(model_name="all-MiniLM-L6-v2", use_model=True)
    dn.build_index(docs)
    dn.save_index(data_dir)
    print(f"  Embeddings saved -> {os.path.join(data_dir, 'dense_embeddings.npy')}")

    print("Index build complete.")
    return data_dir


# Backward-compat alias kept for existing Makefile target
def build_sample_index():
    build_fiqa_index()


if __name__ == "__main__":
    build_fiqa_index()
