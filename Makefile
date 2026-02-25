.PHONY: setup build test test-python test-mojo lint typecheck run clean all

TYPECHECK_PATHS = \
	retrain/type_defs.py \
	retrain/data.py \
	retrain/sepa.py \
	retrain/advantages.py \
	retrain/backend_definitions.py \
	retrain/registry.py \
	retrain/backends.py \
	retrain/trainer.py \
	retrain/prime_rl_backend.py \
	retrain/verifiers_bridge.py

all: setup build

setup:
	uv sync

build:
	mojo build src/main.mojo -o retrain-tinker

# Run both Python and Mojo tests
test: test-python test-mojo

test-python:
	uv run python -m pytest tests/test_*.py -x -q

test-mojo:
	mojo run tests/test_advantages.mojo
	mojo run tests/test_sepa.mojo
	mojo run tests/test_config.mojo
	mojo run tests/test_logging.mojo
	mojo run tests/test_pybridge.mojo
	mojo run tests/test_reward.mojo
	mojo run tests/test_data.mojo
	mojo run tests/test_advantage_fns.mojo
	mojo run tests/test_main.mojo
	mojo run tests/test_backend.mojo
	mojo run tests/test_backpressure.mojo

lint:
	uv run ruff check retrain tests

typecheck:
	uv run ty check $(TYPECHECK_PATHS) --no-progress --output-format concise

run: retrain-tinker
	./retrain-tinker

retrain-tinker: src/*.mojo
	mojo build src/main.mojo -o retrain-tinker

clean:
	rm -f retrain-tinker
	rm -rf retrain.egg-info/
	rm -rf retrain/__pycache__/ retrain/inference_engine/__pycache__/
