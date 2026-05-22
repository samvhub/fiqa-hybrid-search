.PHONY: index eval bench test

index:
	python src/index/indexer.py

eval:
	python src/eval.py

bench: eval

test:
	pytest -q
