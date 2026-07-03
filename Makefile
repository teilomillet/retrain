.PHONY: setup build test test-python test-mojo lint typecheck chaos-backend-workflow run clean all

TYPECHECK_PATHS = \
	retrain/config.py \
	retrain/cli.py \
	retrain/type_defs.py \
	retrain/data.py \
	retrain/accelerators.py \
	retrain/diff.py \
	retrain/json_utils.py \
	retrain/metrics_scan.py \
	retrain/logging_utils.py \
	retrain/progress_exporter.py \
	retrain/campaign.py \
	retrain/delight_campaign_summary.py \
	retrain/sepa.py \
	retrain/advantages.py \
	retrain/planning.py \
	retrain/tinker_runtime.py \
	retrain/tinker_backend.py \
	retrain/squeeze.py \
	retrain/unsloth_runtime.py \
	retrain/unsloth_backend.py \
	retrain/backend_definitions.py \
	retrain/registry.py \
	retrain/backends.py \
	retrain/rewards.py \
	retrain/inference_engine/max_runtime.py \
	retrain/inference_engine/max_engine.py \
	retrain/inference_engine/pytorch_engine.py \
	retrain/inference_engine/openai_engine.py \
	retrain/backpressure.py \
	retrain/fast_lora.py \
	retrain/selective_logprobs.py \
	retrain/policy_loss.py \
	retrain/qwen35_gated_delta.py \
	retrain/local_train_helper.py \
	retrain/flow.py \
	retrain/training_runner.py \
	retrain/tree.py \
	retrain/trainer.py \
	retrain/ttt_discover.py \
	retrain/prime_rl_backend.py \
	retrain/verifiers_bridge.py

VENV_PYTHON = .venv/bin/python
TY ?= uv run ty

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
	$(TY) check $(TYPECHECK_PATHS) --no-progress --output-format concise

chaos-backend-workflow:
	UV_CACHE_DIR=$(CURDIR)/.uv-cache $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),uv run python) scripts/run_backend_workflow_chaos.py

run: retrain-tinker
	./retrain-tinker

retrain-tinker: src/*.mojo
	mojo build src/main.mojo -o retrain-tinker

clean:
	rm -f retrain-tinker
	rm -rf retrain.egg-info/
	rm -rf retrain/__pycache__/ retrain/inference_engine/__pycache__/
