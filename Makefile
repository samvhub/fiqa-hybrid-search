.PHONY: index eval bench test

index:
	python -c "from src.index.indexer import build_sample_index; build_sample_index()"

eval:
	python src/eval.py

bench: eval

test:
	pytest -q
