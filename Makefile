.PHONY: install ingest build features test clean

install:
	pip install -e .

ingest:
	python -m nflproj.cli ingest

build:
	python -m nflproj.cli build

features:
	python -m nflproj.cli features

test:
	pytest tests/ -v

clean:
	rm -rf data/parquet/*.parquet
	rm -rf data/db/*.duckdb
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
