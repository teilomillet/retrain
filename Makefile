.PHONY: setup test lint typecheck chaos-backend-workflow clean all

TYPECHECK_PATHS = retrain
CLEAN_PYTHON_PATHS = retrain scripts tests campaigns experiments
CLEAN_DIRS = build dist site retrain.egg-info .pytest_cache .ruff_cache .hypothesis .ordeal .uv-cache

VENV_PYTHON = .venv/bin/python
TY ?= uv run ty

all: setup

setup:
	uv sync

test:
	uv run python -m pytest tests/ -x -q

lint:
	uv run ruff check retrain scripts tests

typecheck:
	$(TY) check $(TYPECHECK_PATHS) --no-progress --output-format concise

chaos-backend-workflow:
	UV_CACHE_DIR=$(CURDIR)/.uv-cache $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),uv run python) scripts/run_backend_workflow_chaos.py

clean:
	rm -rf $(CLEAN_DIRS) ordeal-report.json
	find $(CLEAN_PYTHON_PATHS) -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name .DS_Store -exec rm -f {} +
