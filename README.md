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

```bash
docker build -t fiqa-search .

# Build index inside container (only needed if data/ not pre-built)
docker run --rm -v "$(pwd)/data:/app/data" fiqa-search python src/index/indexer.py

# Run eval
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" \
    fiqa-search python src/eval.py

# Interactive search
docker run --rm -v "$(pwd)/data:/app/data" fiqa-search \
    --query "what is short selling?" --top-k 10
```

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
