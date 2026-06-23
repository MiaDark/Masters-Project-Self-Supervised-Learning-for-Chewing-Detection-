.PHONY: help setup download mlflow-ui lint format test clean

help:
	@echo "Common commands:"
	@echo "  make setup       Create venv and install dependencies (editable)"
	@echo "  make download    Download raw data from S3 into data/raw"
	@echo "  make mlflow-ui    Launch the MLflow tracking UI at http://127.0.0.1:5000"
	@echo "  make lint        Run ruff checks"
	@echo "  make format      Auto-format with ruff"
	@echo "  make test        Run the test suite"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]" || \
		(. .venv/bin/activate && pip install -r requirements.txt && pip install -e .)
	@echo "Done. Activate with: source .venv/bin/activate"

download:
	python -m chewing_ssl.data.download

mlflow-ui:
	mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-file:./mlruns}

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
