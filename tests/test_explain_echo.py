"""Focused regressions for ECHO and environment dry-run observability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrain.commands.explain.run import run as run_explain


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "echo-openenv.toml"
    config_path.write_text(
        """
[model]
model = "Qwen/Qwen3-4B-Instruct-2507"

[algorithm]
advantage_mode = "grpo"
transform_mode = "none"

[training]
max_steps = 1
batch_size = 2
group_size = 4

[echo]
enabled = true
weight = 0.2
loss_fn = "cross_entropy"
max_tokens_per_step = 1536
max_token_ratio = 0.25
entropy_floor = 0.02
min_prompt_overlap = 0.75

[environment]
provider = "openenv"
id = "http://localhost:8765"
max_turns = 324
args = { renderer = "quaero.render_observation", expected_task_source = "factory-v3", expected_task_ids = ["task-3000", "task-3001"] }
""".strip()
        + "\n"
    )
    return config_path


def test_text_exposes_echo_and_environment_training_contract(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)

    run_explain([str(config_path)])

    output = capsys.readouterr().out
    assert "condition     : grpo+none+echo" in output
    assert "echo          : weight=0.2 loss=cross_entropy" in output
    assert "echo caps     : tokens/step=1536 ratio=0.25 entropy_floor=0.02" in output
    assert "env max_turns : 324" in output
    assert "env renderer  : quaero.render_observation" in output
    assert 'task guard    : source=factory-v3 ids=["task-3000","task-3001"]' in output


def test_json_exposes_echo_and_environment_training_contract(tmp_path, capsys) -> None:
    config_path = _write_config(tmp_path)

    run_explain(["--json", str(config_path)])

    payload = json.loads(capsys.readouterr().out)
    assert payload["condition"] == "grpo+none+echo"
    assert payload["echo_enabled"] is True
    assert payload["echo_weight"] == pytest.approx(0.2)
    assert payload["echo_loss_fn"] == "cross_entropy"
    assert payload["echo_max_tokens_per_step"] == 1536
    assert payload["echo_max_token_ratio"] == pytest.approx(0.25)
    assert payload["echo_entropy_floor"] == pytest.approx(0.02)
    assert payload["echo_min_prompt_overlap"] == pytest.approx(0.75)
    assert payload["environment_provider"] == "openenv"
    assert payload["environment_id"] == "http://localhost:8765"
    assert payload["environment_max_turns"] == 324
    assert payload["environment_renderer"] == "quaero.render_observation"
    assert payload["environment_expected_task_source"] == "factory-v3"
    assert payload["environment_expected_task_ids"] == ["task-3000", "task-3001"]
