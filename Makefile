.PHONY: install install-inference dev lint type test cov run ingest eval docker-build docker-up clean

install:
	pip install -e ".[dev]"

install-inference:
	pip install -e ".[dev,inference]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type:
	mypy

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

run:
	uvicorn rag_qa.api.main:create_app --factory --host 0.0.0.0 --port 8000

ingest:
	rag-ingest --source data/sample_docs

eval:
	rag-eval --dataset eval/datasets/sample_eval.jsonl --report eval/reports/latest.json

docker-build:
	docker build -t rag-document-qa:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
