"""Tests for metrics JSONL scanning."""

from pathlib import Path

from retrain.metrics.scan import iter_jsonl_objects, scan_metrics_file


def test_iter_jsonl_objects_skips_blank_bad_and_non_object_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n"
        '{"step": 1, "loss": 0.5}\n'
        "not json\n"
        "[1, 2, 3]\n"
        '{"step": 2, "correct_rate": 1.0}\n',
        encoding="utf-8",
    )

    assert list(iter_jsonl_objects(metrics_path)) == [
        {"step": 1, "loss": 0.5},
        {"step": 2, "correct_rate": 1.0},
    ]


def test_scan_metrics_file_collects_selected_single_pass_aggregates(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"step": 1, "step_time_s": 2.0, "loss": 0.6, "correct_rate": 0.0}\n'
        '{"step": 2, "time_s": 3.0, "loss": 0.4, "correct_rate": 1.0}\n'
        '{"step": 3, "step_time_s": true, "loss": "bad"}\n',
        encoding="utf-8",
    )

    result = scan_metrics_file(
        metrics_path,
        mean_fields=("loss",),
        max_fields=("correct_rate",),
        collect_step_times=True,
        collect_correct_rates=True,
    )

    assert result.rows == 3
    assert result.last == {"step": 3, "step_time_s": True, "loss": "bad"}
    assert result.wall_time_s == 5.0
    assert result.step_times == [2.0, 3.0]
    assert result.correct_rates == [0.0, 1.0]
    assert result.mean("loss") == 0.5
    assert result.maximum("correct_rate") == 1.0
