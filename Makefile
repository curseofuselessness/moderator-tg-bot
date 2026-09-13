.PHONY: help install install-dev test lint format typecheck clean

help:  ## Show commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:  ## Run tests
	pytest

lint:  ## Check code
	ruff check .

format:  ## Format
	black .
	ruff check --fix .

typecheck:  ## Types check
	mypy bot ml_service shared training

clean:  ## Clean cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage