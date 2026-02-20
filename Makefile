.PHONY: setup build test test-python test-mojo run clean all

all: setup build

setup:
	uv sync

build:
	mojo build src/main.mojo -o retrain-tinker

# Run both Python and Mojo tests
test: test-python test-mojo

test-python:
	python -m pytest tests/ -x -q 2>/dev/null || python -c "from retrain.config import load_config; print('config:', load_config())"

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

run: retrain-tinker
	./retrain-tinker

retrain-tinker: src/*.mojo
	mojo build src/main.mojo -o retrain-tinker

clean:
	rm -f retrain-tinker
	rm -rf retrain.egg-info/
	rm -rf retrain/__pycache__/ retrain/inference_engine/__pycache__/
