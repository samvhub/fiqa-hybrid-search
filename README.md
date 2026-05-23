# FiQA Hybrid Search

Retrieval system over the [BEIR/FiQA](https://github.com/beir-cellar/beir) financial Q&A corpus
(~57 K passages). Implements BM25, dense (all-MiniLM-L6-v2), and hybrid retrieval with a full
evaluation harness.

## Hardware (results measured on)

<!-- Fill in after running eval -->
- **CPU**: Intel Core i7-1165G7 (4 cores / 8 threads, 2.8 GHz base)
- **RAM**: 16 GB
- **OS**: Windows 11 Pro

## Setup

```bash
pip install -r requirements.txt
```

## Build the index (one-time, ~5–20 min on CPU)

Downloads FiQA (~9 MB), encodes all passages with `all-MiniLM-L6-v2`:

```bash
make index
# or: python src/index/indexer.py
```

Pre-built index artifacts are saved to `data/fiqa/`.

## Search

```bash
python src/search.py --query "what is short selling?" --top-k 10
```

Falls back to 3 sample passages if the index has not been built yet.

## Run evaluation

```bash
make bench
# or: python src/eval.py
```

Writes `results/bench.json` (all metrics) and `results/failure_candidates.json`.

## Tests

```bash
pytest -q
```

## Docker

The image is self-contained: `docker build` downloads FiQA (~18 MB), encodes all passages,
and bakes the index + model weights into the image. No internet access is needed at run time.

```bash
# Build (~20 min first time; fast no-op if data/fiqa/ index is already present)
docker build -t fiqa-search .

# Run the full benchmark — writes results/bench.json inside the container
docker run --rm fiqa-search

# Persist results to the host
docker run --rm -v "$(pwd)/results:/app/results" fiqa-search

# Interactive search
docker run --rm fiqa-search python src/search.py --query "what is short selling?" --top-k 10
```

To speed up rebuilds, place the pre-built index files (`data/fiqa/bm25_index.pkl` and
`data/fiqa/dense_embeddings.npy`) in the build context before running `docker build`.
The indexer detects them and skips the encoding step entirely.

## Repository layout

```
├── README.md
├── DESIGN.md             # design doc
├── Dockerfile
├── Makefile
├── src/
│   ├── search.py         # CLI entry point
│   ├── eval.py           # evaluation harness
│   └── retrieval/
│       ├── bm25.py
│       ├── dense.py
│       ├── hybrid.py
│       └── utils.py
├── tests/
│   ├── test_chunking.py
│   ├── test_hybrid.py
│   └── test_eval_metrics.py
├── results/
│   ├── bench.json
│   └── failures.md
└── requirements.txt
```
