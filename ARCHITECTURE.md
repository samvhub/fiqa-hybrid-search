# Architecture — FiQA Hybrid Search

## 1. System Overview

A CPU-only information retrieval system over the BEIR/FiQA financial Q&A corpus (~57K passages).
Supports three retrieval strategies — BM25, Dense, and Hybrid — exposed via a CLI and evaluated
by a full harness that measures quality (Recall@10, MRR) and operational metrics (latency, RAM).

---

## 2. Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE (one-time, run via `make index`)                       │
│                                                                 │
│   BEIR/FiQA                                                     │
│   (remote zip)  ──► indexer.py ──► BM25Retriever.build_index() │
│                           │    └──► DenseRetriever.build_index()│
│                           │                                     │
│                           ▼                                     │
│              data/fiqa/                                         │
│               ├── bm25_index.pkl        (98 MB, gitignored)    │
│               ├── dense_embeddings.npy  (84 MB, gitignored)    │
│               ├── doc_ids.json          (committed)            │
│               ├── dev_queries.json      (committed)            │
│               └── dev_qrels.json        (committed)            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SERVE (query time)                                             │
│                                                                 │
│  search.py (CLI)                                                │
│      │                                                          │
│      ├── BM25Retriever.search()  ──────────────┐               │
│      │        rank-bm25 BM25Okapi               │               │
│      │                                          ▼               │
│      └── DenseRetriever.search() ──► HybridRetriever.search()  │
│               SentenceTransformer                               │
│               + numpy cosine sim                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  EVALUATE (run via `make bench`)                                │
│                                                                 │
│  eval.py                                                        │
│      ├── load_indexes()          load .pkl + .npy               │
│      ├── eval_retriever()        per-query loop, latency timing │
│      ├── ablation_alpha_sweep()  re-run hybrid at α=[0.3,0.5,0.7]│
│      └── collect_failures()     top-10 recall=0 cases          │
│                 │                                               │
│                 ▼                                               │
│          results/bench.json                                     │
│          results/failure_candidates.json                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Indexer — `src/index/indexer.py`

**Responsibility:** Download the corpus once, build both indexes, serialize artifacts.

**Entry point:** `python src/index/indexer.py` (or `make index`)

**Sequence:**
1. `_download_fiqa()` — fetch FiQA zip via HTTPS (`requests`, `verify=False` for Windows SSL),
   extract to `data/fiqa/`. Skips entirely if `corpus.jsonl` is already present.
2. `_load_corpus()` — read `corpus.jsonl` line-by-line → `(doc_ids[], docs[])`.
3. `_load_queries()` / `_load_qrels()` — read `queries.jsonl` and `qrels/dev.tsv`.
4. Serialize `doc_ids.json`, `dev_queries.json`, `dev_qrels.json` (small, committed to git).
5. `BM25Retriever.build_index(docs)` → pickle to `bm25_index.pkl`.
6. `DenseRetriever(use_model=True).build_index(docs)` → save to `dense_embeddings.npy`.

**Skip logic:** If both `bm25_index.pkl` and `dense_embeddings.npy` already exist, the entire
function returns immediately. This makes `docker build` a fast no-op when index artifacts are
pre-built and COPYed into the image.

**Key parameters:**

| Name | Value | Where |
|---|---|---|
| Corpus source | BEIR FiQA (UKP BEIR CDN) | `FIQA_URL` constant |
| Dense model | `all-MiniLM-L6-v2` | hard-coded in `build_fiqa_index()` |
| Batch size (encoding) | 64 | `DenseRetriever.build_index()` |

---

### 3.2 BM25Retriever — `src/retrieval/bm25.py`

**Library:** `rank-bm25` (`BM25Okapi` variant).

**Index:** In-memory inverted index built from whitespace-tokenized documents.

**Public interface:**

```python
bm = BM25Retriever()
bm.build_index(docs: List[str])          # train
results = bm.search(query: str, top_k: int)  # -> List[(doc_idx, score)]
```

**Serialization:** The whole object is pickled (including the underlying BM25Okapi trie and the
`docs` list). At load time the pickle is deserialized in-place — no `build_index` call needed.

**Tokenization:** plain `str.split()` — no stemming, stopword removal, or case normalization.
This is intentional: financial text contains uppercase tickers (AAPL, P/E) where
case-normalization would destroy signal.

**Limitations:** Exact-match only. Fails on paraphrase queries and polysemous terms.

---

### 3.3 DenseRetriever — `src/retrieval/dense.py`

**Library:** `sentence-transformers` (`SentenceTransformer`). TF-IDF fallback when
`sentence-transformers` is unavailable (used in unit tests).

**Index:** `(N, 384)` float32 numpy array of L2-normalized document embeddings.

**Public interface:**

```python
dn = DenseRetriever(model_name="all-MiniLM-L6-v2", use_model=True)

# Path 1 — build from scratch (slow, used by indexer)
dn.build_index(docs: List[str])
dn.save_index(directory: str)    # writes dense_embeddings.npy

# Path 2 — load pre-computed (fast, used by search + eval)
dn.load_index(directory: str, docs: List[str])

# Query time (both paths)
results = dn.search(query: str, top_k: int)  # -> List[(doc_idx, score)]
```

**Query path:** `SentenceTransformer.encode([query])` → `(1, 384)` vector →
`sklearn.cosine_similarity` against the `(57638, 384)` embedding matrix → argsort top-k.

**Memory:** ~84 MB for the embedding matrix + ~88 MB for model weights = ~172 MB incremental
above the BM25 index. Total process RSS after both indexes load: ~1552 MB.

---

### 3.4 HybridRetriever — `src/retrieval/hybrid.py`

**Method:** Min-max score normalization + weighted linear interpolation.

**Algorithm:**
```
bm_scores = BM25Retriever.search(query, top_k * 5)  # wider candidate pool
dn_scores = DenseRetriever.search(query, top_k * 5)

bm_norm = min-max normalize(bm_scores)  # -> [0, 1]
dn_norm = min-max normalize(dn_scores)  # -> [0, 1]

# Union of candidates; missing score = 0
score(doc) = alpha * dn_norm(doc) + (1 - alpha) * bm_norm(doc)

return top_k by combined score
```

**Public interface:**

```python
hy = HybridRetriever(bm, dn, alpha=0.5)
results = hy.search(query: str, top_k: int)  # -> List[(doc_idx, score)]
```

**Alpha semantics:** `alpha=1.0` is pure dense; `alpha=0.0` is pure BM25.
Benchmarked values: 0.3 / 0.5 / 0.7. Best quality at `alpha=0.7` (R@10=0.470),
best latency at pure dense (175ms warm p95 vs 802ms hybrid).

**Latency overhead:** the `top_k * 5` candidate expansion (50 docs each) runs both retrievers
plus two normalization passes and a set-union merge per query, adding ~600ms over dense alone.

---

### 3.5 Search CLI — `src/search.py`

**Entry point:** `python src/search.py --query "..." --top-k 10`

**Startup sequence:**
1. Try to load `data/fiqa/bm25_index.pkl` + `data/fiqa/dense_embeddings.npy`.
2. On success: serve real FiQA corpus.
3. On failure (index not built): print a warning to stderr; fall back to 3 sample passages
   from `src/index/sample_passages.json` with TF-IDF dense (no model download needed).

**Output format:** ranked list — `rank. score=X.XXXX  id=<doc_id>  text=<first 200 chars>`

**Retriever used at serve time:** Hybrid at `alpha=0.5` (default).

---

### 3.6 Evaluation Harness — `src/eval.py`

**Entry point:** `python src/eval.py` (or `make bench`)

**Outputs:**
- `results/bench.json` — all quality and latency metrics (see schema below)
- `results/failure_candidates.json` — raw data for the top-10 recall=0 failure cases

**Metric helpers (also imported by unit tests):**

```python
recall_at_k(retrieved_ids, relevant_ids, k) -> float
mrr(retrieved_ids, relevant_ids) -> float
```

**Latency measurement:**
- Uses `time.perf_counter()` around each `retriever.search()` call.
- Cold = first 20 queries (no warmup).
- Warmup window = queries 21-120 (timed but not reported).
- Warm = queries 121+ (full OS + JIT warmup).

**RAM measurement:** `psutil.Process.memory_info().rss` before and after `load_indexes()`.
Falls back to `tracemalloc` peak if psutil is unavailable.

**bench.json schema:**

```
{
  "bm25":   { recall_at_10, mrr, p50_ms, p95_ms, cold_p50_ms, cold_p95_ms,
              warm_p50_ms, warm_p95_ms, n_queries, stratified, peak_ram_mb },
  "dense":  { ... same ... },
  "hybrid": { ... same ... },
  "ablation": {
    "alpha_0.3": { alpha, recall_at_10, mrr, p50_ms, p95_ms },
    "alpha_0.5": { ... },
    "alpha_0.7": { ... }
  }
}
```

---

### 3.7 Utilities — `src/retrieval/utils.py`

`chunk_text(text, max_tokens, overlap)` — splits text into overlapping token windows.
Currently unused in the main pipeline (FiQA passages are already short); available for
experiments on longer documents (SEC filings, earnings transcripts).

---

## 4. Data Model

### Corpus artifacts (all under `data/fiqa/`)

| File | Size | Committed | Description |
|---|---|---|---|
| `corpus.jsonl` | 46 MB | No | Raw BEIR corpus: `{"_id": ..., "text": ..., "title": ...}` per line |
| `queries.jsonl` | — | No | Raw BEIR queries: `{"_id": ..., "text": ...}` per line |
| `qrels/dev.tsv` | — | No | Relevance judgments: `query_id \t corpus_id \t score` |
| `doc_ids.json` | 553 KB | Yes | Ordered list of doc IDs, index-aligned with `bm.docs` |
| `dev_queries.json` | — | Yes | Dict `{qid: text}` for the 500 dev queries with qrels |
| `dev_qrels.json` | — | Yes | Dict `{qid: {did: score}}` relevance judgments |
| `bm25_index.pkl` | 98 MB | No | Pickled `BM25Retriever` (BM25Okapi trie + `docs` list) |
| `dense_embeddings.npy` | 84 MB | No | `(57638, 384)` float32 array of passage embeddings |

The doc index position is the shared key across all arrays: `doc_ids[i]` is the BEIR document
ID for `bm.docs[i]` and `dn.embeddings[i]`. Retrievers return `(doc_idx, score)` pairs, and
callers use `doc_ids[doc_idx]` to resolve the original document ID.

---

## 5. Execution Paths

### Index build

```
make index
  └── python src/index/indexer.py
        ├── _download_fiqa()          ~18 MB download, ~5 s
        ├── BM25Retriever.build_index() ~10 s
        └── DenseRetriever.build_index()  ~20 min on CPU (57638 passages, batch=64)
```

### Benchmark

```
make bench
  └── python src/eval.py
        ├── load_indexes()            ~10 s (deserialize pkl + load npy + warm ST model)
        ├── eval_retriever(bm25)      ~3 min (500 queries × ~360 ms/query)
        ├── eval_retriever(dense)     ~1.5 min (500 queries × ~150 ms/query)
        ├── eval_retriever(hybrid)    ~7 min (500 queries × ~800 ms/query)
        ├── ablation_alpha_sweep()    ~21 min (3 alphas × hybrid timing above)
        └── collect_failures()        <1 s
```

### Docker (end-to-end)

```
docker build -t fiqa-search .
  ├── pip install requirements.txt
  ├── SentenceTransformer('all-MiniLM-L6-v2')   bake model weights into image
  └── python src/index/indexer.py               build index (or fast no-op if pre-built)

docker run --rm -v "$(pwd)/results:/app/results" fiqa-search
  └── make bench   → writes /app/results/bench.json
```

---

## 6. External Dependencies

| Package | Version pin | Used for |
|---|---|---|
| `rank-bm25` | any | BM25Okapi tokenized inverted index |
| `sentence-transformers` | >=2.2.2 | all-MiniLM-L6-v2 encoding and model loading |
| `numpy` | any | Embedding matrix storage, cosine similarity, percentile stats |
| `scikit-learn` | any | `cosine_similarity`, TF-IDF fallback vectorizer |
| `scipy` | any | Transitive dep of scikit-learn |
| `requests` | any | FiQA zip download with streaming + SSL bypass |
| `psutil` | any | Process RSS measurement for peak RAM |
| `tqdm` | any | Progress bar during dense encoding |

---

## 7. Extension Points

| What to change | Where |
|---|---|
| Swap dense model | `DenseRetriever.__init__` default + `indexer.py` constant |
| Change hybrid fusion method (e.g. RRF) | `HybridRetriever.search()` |
| Add FAISS ANN index | `DenseRetriever.search()` — replace `cosine_similarity` call |
| Add query expansion / HyDE | `search.py` — preprocess `args.query` before passing to retriever |
| Add cross-encoder re-ranking | `search.py` — add re-rank step after `hy.search()` |
| Add chunking for longer docs | `indexer.py` — apply `utils.chunk_text()` before `build_index()` |
| Change eval split (test vs dev) | `indexer.py` `_load_qrels(split="test")` |
