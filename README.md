FiQA Hybrid Search (sample scaffold)

This repository is a scaffold for the Take-Home: Constrained Retrieval System assignment. It contains a minimal, runnable project layout and basic implementations for BM25, dense (fallback), and hybrid retrievers plus evaluation stubs and tests.

See `Makefile` for common commands.

Quick start:

```bash
python src/search.py --query "what is short selling?" --top-k 5
```

To run unit tests:

```bash
pip install -r requirements.txt
pytest -q
```
