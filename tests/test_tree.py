"""Tests for the experiment tech tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrain.tree import (
    Annotation,
    NodeState,
    SuccessCondition,
    Tree,
    TreeNode,
    TreeState,
    _build_parent_map,
    _load_state,
    _state_path,
    effective_status,
    evaluate_node,
    format_next,
    format_show,
    format_tree,
    format_tree_json,
    load_tree,
    parse_success_condition,
    reset_node,
    save_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_TREE = """\
[tree]
name = "test-tree"
description = "A test tree"

[[tree.nodes]]
id = "a"
label = "Node A"
campaign = "campaigns/a.toml"
success = "correct_rate > 0.55"
on_success = ["b"]
on_failure = ["c"]

[[tree.nodes]]
id = "b"
label = "Node B"
campaign = "campaigns/b.toml"
success = "correct_rate > 0.60"

[[tree.nodes]]
id = "c"
label = "Node C"
campaign = "campaigns/c.toml"
"""


def _write_tree(tmp_path: Path, content: str = _MINIMAL_TREE) -> Path:
    p = tmp_path / "tree.toml"
    p.write_text(content)
    return p


def _write_state(tmp_path: Path, state: dict) -> Path:
    p = tmp_path / "tree_state.json"
    p.write_text(json.dumps(state))
    return p


def _make_campaign_dir(
    tmp_path: Path,
    runs: list[dict],
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Path:
    """Create a minimal campaign directory with manifest and metrics."""
    campaign_dir = tmp_path / "campaign_001"
    campaign_dir.mkdir(exist_ok=True)

    if conditions is None:
        conditions = ["grpo+none"]
    if seeds is None:
        seeds = [42]

    run_defs = []
    for run_info in runs:
        name = run_info.get("run_name", "grpo+none_s42")
        run_dir = campaign_dir / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = run_info.get("metrics", [])
        lines = [json.dumps(m) for m in metrics]
        (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n")

        state = run_info.get("trainer_state", {"checkpoint_name": "final", "step": 49})
        (run_dir / "trainer_state.json").write_text(json.dumps(state))

        run_defs.append({
            "run_name": name,
            "log_dir": str(run_dir),
            "condition": run_info.get("condition", "grpo+none"),
            "seed": run_info.get("seed", 42),
        })

    manifest = {
        "timestamp": "20260228_100000",
        "campaign_toml": "test.toml",
        "conditions": conditions,
        "seeds": seeds,
        "max_steps": 50,
        "num_runs": len(runs),
        "runner_pid": 0,
        "runs": run_defs,
    }
    (campaign_dir / "manifest.json").write_text(json.dumps(manifest))
    return campaign_dir


# ---------------------------------------------------------------------------
# TestLoadTree
# ---------------------------------------------------------------------------


class TestLoadTree:
    def test_minimal(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        assert tree.name == "test-tree"
        assert len(tree.nodes) == 3
        assert set(tree.node_map.keys()) == {"a", "b", "c"}
        assert tree.node_map["a"].on_success == ["b"]
        assert tree.node_map["a"].on_failure == ["c"]

    def test_success_conditions_parsed(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        sc = tree.node_map["a"].success_condition
        assert sc is not None
        assert sc.metric == "correct_rate"
        assert sc.op == ">"
        assert sc.threshold == 0.55

    def test_no_success_condition(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        assert tree.node_map["c"].success_condition is None

    def test_missing_id_raises(self, tmp_path):
        toml = """\
[tree]
name = "bad"
[[tree.nodes]]
label = "No ID"
campaign = "x.toml"
"""
        with pytest.raises(ValueError, match="missing 'id'"):
            load_tree(_write_tree(tmp_path, toml))

    def test_missing_campaign_raises(self, tmp_path):
        toml = """\
[tree]
name = "bad"
[[tree.nodes]]
id = "x"
label = "No campaign"
"""
        with pytest.raises(ValueError, match="missing 'campaign'"):
            load_tree(_write_tree(tmp_path, toml))

    def test_bad_child_ref_raises(self, tmp_path):
        toml = """\
[tree]
name = "bad"
[[tree.nodes]]
id = "a"
label = "A"
campaign = "a.toml"
on_success = ["nonexistent"]
"""
        with pytest.raises(ValueError, match="unknown child"):
            load_tree(_write_tree(tmp_path, toml))

    def test_unknown_metric_raises(self, tmp_path):
        toml = """\
[tree]
name = "bad"
[[tree.nodes]]
id = "a"
label = "A"
campaign = "a.toml"
success = "bogus_metric > 0.5"
"""
        with pytest.raises(ValueError, match="unknown metric"):
            load_tree(_write_tree(tmp_path, toml))


# ---------------------------------------------------------------------------
# TestSuccessCondition
# ---------------------------------------------------------------------------


class TestSuccessCondition:
    def test_gt(self):
        sc = parse_success_condition("correct_rate > 0.55")
        assert sc.metric == "correct_rate"
        assert sc.op == ">"
        assert sc.threshold == 0.55
        assert sc.evaluate(0.56) is True
        assert sc.evaluate(0.55) is False

    def test_gte(self):
        sc = parse_success_condition("loss >= 0.1")
        assert sc.evaluate(0.1) is True
        assert sc.evaluate(0.09) is False

    def test_lte(self):
        sc = parse_success_condition("loss <= 0.5")
        assert sc.evaluate(0.5) is True
        assert sc.evaluate(0.51) is False

    def test_eq(self):
        sc = parse_success_condition("mean_reward == 1.0")
        assert sc.evaluate(1.0) is True
        assert sc.evaluate(0.99) is False

    def test_lt(self):
        sc = parse_success_condition("loss < 0.3")
        assert sc.evaluate(0.29) is True
        assert sc.evaluate(0.3) is False

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid success condition"):
            parse_success_condition("not a condition")

    def test_invalid_no_spaces(self):
        # Still valid — regex handles optional whitespace
        sc = parse_success_condition("loss>0.5")
        assert sc.op == ">"


# ---------------------------------------------------------------------------
# TestStateRoundtrip
# ---------------------------------------------------------------------------


class TestStateRoundtrip:
    def test_save_and_reload(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)

        # Add some state
        tree.state.nodes["a"] = NodeState(
            status="done",
            outcome="success",
            campaign_dir="logs/test",
            result={"correct_rate": 0.62},
            annotations=[Annotation(text="it worked", at="2026-02-28T14:00:00")],
        )
        save_state(tree)

        # Reload
        tree2 = load_tree(tree_path)
        ns = tree2.state.nodes["a"]
        assert ns.status == "done"
        assert ns.outcome == "success"
        assert ns.campaign_dir == "logs/test"
        assert ns.result == {"correct_rate": 0.62}
        assert len(ns.annotations) == 1
        assert ns.annotations[0].text == "it worked"

    def test_missing_state_file(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        assert tree.state.nodes == {}

    def test_corrupt_state_file(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        _write_state(tmp_path, "not json {{{")  # type: ignore[arg-type]
        # Write raw invalid JSON
        (tmp_path / "tree_state.json").write_text("not json {{{")
        tree = load_tree(tree_path)
        assert tree.state.nodes == {}

    def test_atomic_write(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(status="running")
        save_state(tree)

        sp = _state_path(tree_path)
        assert sp.exists()
        data = json.loads(sp.read_text())
        assert data["nodes"]["a"]["status"] == "running"


# ---------------------------------------------------------------------------
# TestStatusDerivation
# ---------------------------------------------------------------------------


class TestStatusDerivation:
    def test_root_pending(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        assert effective_status("a", tree) == "pending"

    def test_child_locked(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        # b is child of a via on_success, a not done → locked
        assert effective_status("b", tree) == "locked"

    def test_child_unlocked_on_success(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        assert effective_status("b", tree) == "pending"

    def test_wrong_branch_skipped(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        # a succeeds → c (on_failure) should be skipped
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        assert effective_status("c", tree) == "skipped"

    def test_failure_unlocks_failure_branch(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="failure")
        assert effective_status("c", tree) == "pending"
        assert effective_status("b", tree) == "skipped"

    def test_running_stays_running(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="running")
        assert effective_status("a", tree) == "running"

    def test_done_stays_done(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        assert effective_status("a", tree) == "done"


# ---------------------------------------------------------------------------
# TestEvaluateNode
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    def test_success(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)

        campaign_dir = _make_campaign_dir(
            tmp_path,
            runs=[{
                "run_name": "grpo+none_s42",
                "condition": "grpo+none",
                "seed": 42,
                "metrics": [
                    {"step": 0, "correct_rate": 0.5, "loss": 0.1, "mean_reward": 0.5, "condition": "grpo+none"},
                    {"step": 50, "correct_rate": 0.60, "loss": 0.05, "mean_reward": 0.6, "condition": "grpo+none"},
                ],
                "trainer_state": {"checkpoint_name": "final", "step": 49},
            }],
        )

        tree.state.nodes["a"] = NodeState(
            status="done",
            campaign_dir=str(campaign_dir),
        )

        result = evaluate_node(tree, "a")
        assert result == "success"
        assert tree.state.nodes["a"].outcome == "success"
        assert tree.state.nodes["a"].result["correct_rate"] == 0.6

    def test_failure(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)

        campaign_dir = _make_campaign_dir(
            tmp_path,
            runs=[{
                "run_name": "grpo+none_s42",
                "condition": "grpo+none",
                "seed": 42,
                "metrics": [
                    {"step": 50, "correct_rate": 0.40, "loss": 0.1, "mean_reward": 0.4, "condition": "grpo+none"},
                ],
                "trainer_state": {"checkpoint_name": "final", "step": 49},
            }],
        )

        tree.state.nodes["a"] = NodeState(
            status="done",
            campaign_dir=str(campaign_dir),
        )

        result = evaluate_node(tree, "a")
        assert result == "failure"
        assert tree.state.nodes["a"].outcome == "failure"

    def test_no_condition_returns_none(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["c"] = NodeState(status="done", campaign_dir="x")
        result = evaluate_node(tree, "c")
        assert result is None

    def test_averages_across_seeds(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)

        campaign_dir = _make_campaign_dir(
            tmp_path,
            runs=[
                {
                    "run_name": "grpo+none_s42",
                    "condition": "grpo+none",
                    "seed": 42,
                    "metrics": [
                        {"step": 50, "correct_rate": 0.70, "loss": 0.05, "mean_reward": 0.7, "condition": "grpo+none"},
                    ],
                    "trainer_state": {"checkpoint_name": "final", "step": 49},
                },
                {
                    "run_name": "grpo+none_s101",
                    "condition": "grpo+none",
                    "seed": 101,
                    "metrics": [
                        {"step": 50, "correct_rate": 0.50, "loss": 0.1, "mean_reward": 0.5, "condition": "grpo+none"},
                    ],
                    "trainer_state": {"checkpoint_name": "final", "step": 49},
                },
            ],
            seeds=[42, 101],
        )

        tree.state.nodes["a"] = NodeState(
            status="done",
            campaign_dir=str(campaign_dir),
        )

        result = evaluate_node(tree, "a")
        # Average: (0.70 + 0.50) / 2 = 0.60 — NOT > 0.55, so success
        assert result == "success"
        assert tree.state.nodes["a"].result["correct_rate"] == 0.6

    def test_no_campaign_dir_raises(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(status="done")
        with pytest.raises(ValueError, match="no campaign_dir"):
            evaluate_node(tree, "a")


# ---------------------------------------------------------------------------
# TestFormatTree
# ---------------------------------------------------------------------------


class TestFormatTree:
    def test_status_icons(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        output = format_tree(tree)
        # Root 'a' is pending
        assert "[ ] a" in output
        # Children locked
        assert "[#] b" in output
        assert "[#] c" in output

    def test_branch_display(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        output = format_tree(tree)
        assert "success ->" in output
        assert "failure ->" in output

    def test_done_with_result(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(
            status="done",
            outcome="success",
            result={"correct_rate": 0.62},
        )
        output = format_tree(tree)
        assert "[✓] a" in output
        assert "correct_rate=0.62" in output

    def test_diamond_no_infinite_loop(self, tmp_path):
        """Diamond: a→b, a→c, b→d, c→d — should not loop."""
        toml = """\
[tree]
name = "diamond"

[[tree.nodes]]
id = "a"
label = "A"
campaign = "a.toml"
on_success = ["b", "c"]

[[tree.nodes]]
id = "b"
label = "B"
campaign = "b.toml"
on_success = ["d"]

[[tree.nodes]]
id = "c"
label = "C"
campaign = "c.toml"
on_success = ["d"]

[[tree.nodes]]
id = "d"
label = "D"
campaign = "d.toml"
"""
        tree = load_tree(_write_tree(tmp_path, toml))
        output = format_tree(tree)
        # d should appear once normally and once as "(see above)"
        assert "d" in output
        assert "(see above)" in output

    def test_skipped_icon(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        output = format_tree(tree)
        # c is on_failure of a, which succeeded → skipped
        assert "[-] c" in output

    def test_running_icon(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="running")
        output = format_tree(tree)
        assert "[~] a" in output


# ---------------------------------------------------------------------------
# TestFormatNext
# ---------------------------------------------------------------------------


class TestFormatNext:
    def test_root_pending(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        output = format_next(tree)
        assert "a:" in output
        assert "campaigns/a.toml" in output

    def test_no_pending(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="running")
        tree.state.nodes["b"] = NodeState(status="skipped")
        tree.state.nodes["c"] = NodeState(status="skipped")
        output = format_next(tree)
        assert "No pending" in output

    def test_unlocked_after_success(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        output = format_next(tree)
        assert "b:" in output
        # c is skipped (wrong branch)
        assert "c:" not in output


# ---------------------------------------------------------------------------
# TestCliTree
# ---------------------------------------------------------------------------


class TestCliTree:
    def test_view(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree([str(tree_path)])
        out = capsys.readouterr().out
        assert "test-tree" in out
        assert "[ ] a" in out

    def test_next(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree(["next", str(tree_path)])
        out = capsys.readouterr().out
        assert "a:" in out

    def test_note(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree(["note", "a", "test note", str(tree_path)])
        out = capsys.readouterr().out
        assert "Note added" in out

        # Verify persisted
        state_file = tmp_path / "tree_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["nodes"]["a"]["annotations"][0]["text"] == "test note"

    def test_eval_no_nodes(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree(["eval", str(tree_path)])
        out = capsys.readouterr().out
        assert "No nodes to evaluate" in out

    def test_unknown_node_note(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        with pytest.raises(SystemExit):
            _run_tree(["note", "nonexistent", "text", str(tree_path)])

    def test_unknown_node_run(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        with pytest.raises(SystemExit):
            _run_tree(["run", "nonexistent", str(tree_path)])

    def test_help(self, capsys):
        from retrain.cli import _run_tree

        _run_tree(["--help"])
        out = capsys.readouterr().out
        assert "Usage" in out


# ---------------------------------------------------------------------------
# TestFormatShow
# ---------------------------------------------------------------------------


class TestFormatShow:
    def test_basic(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        output = format_show(tree, "a")
        assert 'a: "Node A"' in output
        assert "campaigns/a.toml" in output
        assert "correct_rate > 0.55" in output
        assert "on_success:" in output

    def test_with_annotations(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(
            annotations=[Annotation(text="Test note", at="2026-02-28T14:00:00")],
        )
        output = format_show(tree, "a")
        assert "annotations:" in output
        assert "[2026-02-28T14:00:00] Test note" in output

    def test_with_result(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(
            status="done",
            outcome="success",
            campaign_dir="logs/test",
            result={"correct_rate": 0.62},
        )
        output = format_show(tree, "a")
        assert "campaign_dir: logs/test" in output
        assert "correct_rate=0.62" in output

    def test_unknown_node_raises(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        with pytest.raises(KeyError, match="Unknown node"):
            format_show(tree, "nonexistent")


# ---------------------------------------------------------------------------
# TestFormatTreeJson
# ---------------------------------------------------------------------------


class TestFormatTreeJson:
    def test_structure(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        data = format_tree_json(tree)
        assert data["name"] == "test-tree"
        assert data["description"] == "A test tree"
        assert len(data["nodes"]) == 3
        ids = [n["id"] for n in data["nodes"]]
        assert ids == ["a", "b", "c"]

    def test_effective_statuses(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        data = format_tree_json(tree)
        by_id = {n["id"]: n for n in data["nodes"]}
        assert by_id["a"]["status"] == "done"
        assert by_id["a"]["outcome"] == "success"
        assert by_id["b"]["status"] == "pending"
        assert by_id["c"]["status"] == "skipped"

    def test_annotations_serialized(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        tree.state.nodes["a"] = NodeState(
            annotations=[Annotation(text="note1", at="2026-01-01T00:00:00")],
        )
        data = format_tree_json(tree)
        by_id = {n["id"]: n for n in data["nodes"]}
        assert len(by_id["a"]["annotations"]) == 1
        assert by_id["a"]["annotations"][0]["text"] == "note1"

    def test_json_serializable(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        data = format_tree_json(tree)
        # Must not raise
        json.dumps(data)


# ---------------------------------------------------------------------------
# TestResetNode
# ---------------------------------------------------------------------------


class TestResetNode:
    def test_resets_fields(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(
            status="done",
            outcome="success",
            campaign_dir="logs/test",
            result={"correct_rate": 0.62},
            started_at="2026-02-28T10:00:00",
            completed_at="2026-02-28T11:00:00",
            annotations=[Annotation(text="keep me", at="2026-02-28T12:00:00")],
        )
        save_state(tree)

        reset_node(tree, "a")
        ns = tree.state.nodes["a"]
        assert ns.status == "pending"
        assert ns.outcome == ""
        assert ns.campaign_dir == ""
        assert ns.result == {}
        assert ns.started_at == ""
        assert ns.completed_at == ""

    def test_preserves_annotations(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(
            status="done",
            outcome="failure",
            annotations=[Annotation(text="important", at="2026-02-28T12:00:00")],
        )
        save_state(tree)

        reset_node(tree, "a")
        assert len(tree.state.nodes["a"].annotations) == 1
        assert tree.state.nodes["a"].annotations[0].text == "important"

    def test_saves_to_disk(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        save_state(tree)

        reset_node(tree, "a")

        # Reload from disk
        tree2 = load_tree(tree_path)
        assert tree2.state.nodes["a"].status == "pending"

    def test_unknown_node_raises(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        with pytest.raises(KeyError, match="Unknown node"):
            reset_node(tree, "nonexistent")


# ---------------------------------------------------------------------------
# TestFormatNextRicher
# ---------------------------------------------------------------------------


class TestFormatNextRicher:
    def test_label_and_notes_appear(self, tmp_path):
        toml = """\
[tree]
name = "test"

[[tree.nodes]]
id = "a"
label = "Node A label"
campaign = "campaigns/a.toml"
notes = "Some important note"
"""
        tree = load_tree(_write_tree(tmp_path, toml))
        output = format_next(tree)
        assert "a: campaigns/a.toml" in output
        assert '"Node A label"' in output
        assert "Some important note" in output

    def test_label_without_notes(self, tmp_path):
        tree = load_tree(_write_tree(tmp_path))
        output = format_next(tree)
        # Should still show label even without notes
        assert '"Node A"' in output


# ---------------------------------------------------------------------------
# TestCliTreeShow
# ---------------------------------------------------------------------------


class TestCliTreeShow:
    def test_show(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree(["show", "a", str(tree_path)])
        out = capsys.readouterr().out
        assert "Node A" in out
        assert "campaigns/a.toml" in out

    def test_show_unknown_node(self, tmp_path):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        with pytest.raises(SystemExit):
            _run_tree(["show", "nonexistent", str(tree_path)])


# ---------------------------------------------------------------------------
# TestCliTreeReset
# ---------------------------------------------------------------------------


class TestCliTreeReset:
    def test_reset(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        # Set up state first
        tree = load_tree(tree_path)
        tree.state.nodes["a"] = NodeState(status="done", outcome="success")
        save_state(tree)

        _run_tree(["reset", "a", str(tree_path)])
        out = capsys.readouterr().out
        assert "reset to pending" in out

        # Verify on disk
        tree2 = load_tree(tree_path)
        assert tree2.state.nodes["a"].status == "pending"


# ---------------------------------------------------------------------------
# TestCliTreeJson
# ---------------------------------------------------------------------------


class TestCliTreeJson:
    def test_view_json(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree([str(tree_path), "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["name"] == "test-tree"
        assert len(data["nodes"]) == 3

    def test_show_json(self, tmp_path, capsys):
        tree_path = _write_tree(tmp_path)
        from retrain.cli import _run_tree

        _run_tree(["show", "a", str(tree_path), "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "a"
        assert data["campaign"] == "campaigns/a.toml"
