.PHONY: setup test lint typecheck chaos-backend-workflow clean all

TYPECHECK_PATHS = retrain

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
	rm -rf retrain.egg-info/
	rm -rf retrain/__pycache__/ retrain/inference_engine/__pycache__/
