from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrain.config import TrainConfig
from retrain.training.echo import (
    EchoBuildStats,
    EchoLimitStats,
    assert_echo_live_observation_contract,
    build_rollout_echo_datum,
    build_rollout_echo_datums,
    common_prefix_len,
    limit_echo_masks,
    merge_echo_build_stats,
    retain_all_echo_masks,
    run_rl_echo_train_step,
)
from retrain.training.rollout.state import RolloutAccumulator, prepare_echo_step_plan


def test_strict_echo_live_observation_contract_accepts_real_tool_signal() -> None:
    assert_echo_live_observation_contract(
        required=True,
        build=EchoBuildStats(
            candidate_datums=2,
            candidate_tokens=3,
            observation_mask_datums=2,
            observation_responses=2,
            executed_transition_datums=2,
            bridged_transition_datums=2,
            explicit_transition_rollouts=1,
        ),
        limit=EchoLimitStats(kept_datums=2, kept_tokens=3),
        final_masks=[[0.0, 0.1, 0.1], [0.1]],
        eligible_rollouts=1,
        skipped_entropy_floor=False,
    )


@pytest.mark.parametrize(
    ("build", "limit", "masks", "failure"),
    [
        (
            EchoBuildStats(
                candidate_tokens=3,
                observation_responses=1,
                bridged_transition_datums=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(kept_tokens=2),
            [[0.1, 0.1]],
            "all_candidate_tokens_retained",
        ),
        (
            EchoBuildStats(
                candidate_tokens=2,
                observation_responses=1,
                bridged_transition_datums=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(kept_tokens=2, truncated_tokens=1),
            [[0.1, 0.1]],
            "zero_truncated_tokens",
        ),
        (
            EchoBuildStats(
                candidate_tokens=2,
                terminal_candidate_tokens=1,
                observation_responses=1,
                bridged_transition_datums=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(kept_tokens=2, kept_terminal_tokens=0),
            [[0.1, 0.1]],
            "all_terminal_tokens_retained",
        ),
    ],
)
def test_strict_echo_full_retention_contract_fails_on_information_loss(
    build, limit, masks, failure
) -> None:
    with pytest.raises(RuntimeError, match=failure):
        assert_echo_live_observation_contract(
            required=True,
            build=build,
            limit=limit,
            final_masks=masks,
            eligible_rollouts=1,
            skipped_entropy_floor=False,
            target_retention="all",
        )


def test_strict_echo_bounded_retention_allows_declared_truncation() -> None:
    assert_echo_live_observation_contract(
        required=True,
        build=EchoBuildStats(
            candidate_tokens=3,
            terminal_candidate_tokens=1,
            observation_responses=1,
            executed_transition_datums=1,
            bridged_transition_datums=1,
            explicit_transition_rollouts=1,
        ),
        limit=EchoLimitStats(kept_tokens=2, truncated_tokens=1),
        final_masks=[[0.1, 0.1]],
        eligible_rollouts=1,
        skipped_entropy_floor=False,
        target_retention="bounded",
    )


def test_strict_echo_full_retention_rejects_missing_transition_response() -> None:
    with pytest.raises(RuntimeError, match="all_executed_transitions_observed"):
        assert_echo_live_observation_contract(
            required=True,
            build=EchoBuildStats(
                candidate_datums=1,
                candidate_tokens=1,
                observation_mask_datums=1,
                observation_responses=1,
                executed_transition_datums=2,
                bridged_transition_datums=1,
                explicit_transition_rollouts=2,
            ),
            limit=EchoLimitStats(kept_datums=1, kept_tokens=1),
            final_masks=[[0.1]],
            eligible_rollouts=2,
            skipped_entropy_floor=False,
            target_retention="all",
        )


@pytest.mark.parametrize(
    ("build", "limit", "masks", "eligible", "skipped", "failure"),
    [
        (EchoBuildStats(), EchoLimitStats(), [], 0, False, "candidate_tokens"),
        (
            EchoBuildStats(
                candidate_tokens=1,
                observation_responses=1,
                bridged_transition_datums=1,
                bridge_failures=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(kept_tokens=1),
            [[0.1]],
            1,
            False,
            "zero_bridge_failures",
        ),
        (
            EchoBuildStats(
                candidate_tokens=1,
                observation_responses=1,
                bridged_transition_datums=1,
                renderer_parity_failures=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(kept_tokens=1),
            [[0.1]],
            1,
            False,
            "zero_renderer_parity_failures",
        ),
        (
            EchoBuildStats(
                candidate_tokens=1,
                observation_responses=1,
                bridged_transition_datums=1,
                explicit_transition_rollouts=1,
            ),
            EchoLimitStats(),
            [[0.0]],
            1,
            True,
            "kept_tokens",
        ),
    ],
)
def test_strict_echo_live_observation_contract_fails_closed(
    build, limit, masks, eligible, skipped, failure
) -> None:
    with pytest.raises(RuntimeError, match=failure):
        assert_echo_live_observation_contract(
            required=True,
            build=build,
            limit=limit,
            final_masks=masks,
            eligible_rollouts=eligible,
            skipped_entropy_floor=skipped,
        )


def test_common_prefix_len_stops_at_first_mismatch() -> None:
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2


def test_build_rollout_echo_datum_interleaves_actions_and_observations() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50, 51, 99],
            completion_ids=[4, 5],
            completion_logprobs=[-0.2, -0.3],
            observation_mask=[0, 0, 0, 1, 1, 0],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [-0.2, 0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.tokens == [1, 2, 3, 50, 51, 99, 4, 5]
    assert datum.logprobs == [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -0.2, -0.3]
    assert datum.advantages == [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, -0.2, 0.4]
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0]
    assert datum.full_observation_count == 2
    assert datum.positive_tokens == 2
    assert datum.action_token_count == 3
    assert datum.action_surprisal_sum == pytest.approx(0.6)
    assert stats.candidate_datums == 1
    assert stats.observation_mask_datums == 1
    assert stats.skipped_first_turns == 1


def test_build_rollout_echo_datum_rejects_unstable_prompt_stitching() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[9, 8, 7],
            completion_ids=[4],
            completion_logprobs=[-0.2],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.1], [0.2]],
        weight=0.05,
        min_prompt_overlap=0.5,
    )

    assert datum is None
    assert stats.skipped_low_overlap == 1


def test_build_rollout_echo_datum_keeps_exact_prefix_before_late_mismatch() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50],
            completion_ids=[4],
            completion_logprobs=[-0.2],
            observation_mask=[0, 0, 0, 1],
        ),
        SimpleNamespace(
            prompt_ids=[9, 8, 7],
            completion_ids=[5],
            completion_logprobs=[-0.3],
        ),
    ]

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=[[0.1], [0.2], [0.3]],
        weight=0.05,
        min_prompt_overlap=0.5,
    )

    assert [datum.tokens for datum in datums] == [
        [1, 2, 3, 50, 4],
        [9, 8, 7, 5],
    ]
    assert [datum.action_token_count for datum in datums] == [2, 1]
    assert sum(datum.action_token_count for datum in datums) == 3
    assert datums[0].advantages == [0.0, 0.0, 0.1, 0.0, 0.2]
    assert datums[0].echo_advantages == [0.0, 0.0, 0.0, 0.05, 0.0]
    assert datums[0].positive_tokens == 1
    assert datums[1].full_observation_count == 1
    assert stats.candidate_datums == 1
    assert stats.candidate_tokens == 1
    assert stats.skipped_low_overlap == 1
    assert stats.split_non_prefix == 1


def test_echo_high_overlap_non_prefix_splits_instead_of_splicing() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-0.1, -0.2],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 9, 50],
            completion_ids=[5],
            completion_logprobs=[-0.3],
        ),
    ]

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=[[0.1, 0.2], [0.3]],
        weight=0.05,
        min_prompt_overlap=0.5,
    )

    assert [datum.tokens for datum in datums] == [
        [1, 2, 3, 4],
        [1, 2, 3, 9, 50, 5],
    ]
    assert sum(datum.action_token_count for datum in datums) == 3
    assert stats.split_non_prefix == 1
    assert stats.skipped_low_overlap == 0


def test_echo_shorter_next_prompt_splits_and_keeps_every_action() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2, 3],
            completion_ids=[4],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[5],
            completion_logprobs=[-0.2],
        ),
    ]

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=[[0.1], [0.2]],
        weight=0.05,
        min_prompt_overlap=0.5,
    )

    assert [datum.tokens for datum in datums] == [[1, 2, 3, 4], [1, 2, 5]]
    assert sum(datum.action_token_count for datum in datums) == 2
    assert stats.split_non_prefix == 1


def test_explicit_transition_rows_keep_actions_and_target_each_response_once() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
            echo_observation_capture_supported=True,
            post_observation_ids=[1, 2, 3, 40, 50],
            post_observation_mask=[0, 0, 0, 0, 1],
            post_observation_seen=True,
            post_observation_terminal=False,
        ),
        # Deliberately not a prefix of the prior transition. This is the real
        # Qwen rerender shape that defeated rollout-level suffix stitching.
        SimpleNamespace(
            prompt_ids=[9, 8],
            completion_ids=[4, 5],
            completion_logprobs=[-0.2, -0.3],
            echo_observation_capture_supported=True,
            post_observation_ids=[9, 8, 4, 5, 41, 51, 52],
            post_observation_mask=[0, 0, 0, 0, 0, 1, 1],
            post_observation_seen=True,
            post_observation_terminal=True,
        ),
    ]

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=[[0.7], [-0.2, 0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert [datum.tokens for datum in datums] == [
        [1, 2, 3, 40, 50],
        [9, 8, 4, 5, 41, 51, 52],
    ]
    assert sum(datum.action_token_count for datum in datums) == 3
    assert datums[0].advantages == [0.0, 0.0, 0.7, 0.0, 0.0]
    assert datums[1].advantages == [0.0, 0.0, -0.2, 0.4, 0.0, 0.0, 0.0]
    assert datums[0].echo_advantages == [0.0, 0.0, 0.0, 0.0, 0.05]
    assert datums[1].echo_advantages == [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05]
    assert datums[0].terminal_observation_mask == [0, 0, 0, 0, 0]
    assert datums[1].terminal_observation_mask == [0, 0, 0, 0, 0, 1, 1]
    assert all(datum.full_observation_count == 3 for datum in datums)
    assert stats.observation_responses == 2
    assert stats.bridged_transition_datums == 2
    assert stats.bridge_failures == 0
    assert stats.terminal_candidate_tokens == 2
    assert stats.explicit_transition_rollouts == 1
    assert stats.split_non_prefix == 0

    _, capped = limit_echo_masks(
        [datum.echo_advantages for datum in datums],
        max_positive_tokens=1,
        terminal_observation_masks=[
            datum.terminal_observation_mask for datum in datums
        ],
    )
    assert capped.kept_tokens == 1
    assert capped.kept_terminal_tokens == 0


def test_explicit_bridge_failure_preserves_grpo_action_row() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            completion_logprobs=[-0.1, -0.2],
            echo_observation_capture_supported=True,
            post_observation_ids=None,
            post_observation_mask=None,
            post_observation_seen=True,
            post_observation_bridge_failed=True,
            post_observation_terminal=True,
        )
    ]

    datums, stats = build_rollout_echo_datums(
        turns,
        completion_advantages=[[0.25, -0.25]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert len(datums) == 1
    assert datums[0].tokens == [1, 2, 3, 4]
    assert datums[0].advantages == [0.0, 0.0, 0.25, -0.25]
    assert datums[0].action_token_count == 2
    assert datums[0].positive_tokens == 0
    assert stats.observation_responses == 1
    assert stats.bridge_failures == 1
    assert stats.bridged_transition_datums == 0


def test_build_rollout_echo_datum_falls_back_to_prompt_suffix() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50, 51],
            completion_ids=[4],
            completion_logprobs=[-0.2],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.tokens == [1, 2, 3, 50, 51, 4]
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.05, 0.05, 0.0]
    assert datum.full_observation_count == 2
    assert datum.positive_tokens == 2
    assert stats.candidate_datums == 1
    assert stats.observation_mask_datums == 0


def test_build_rollout_echo_datum_skips_bad_observation_mask() -> None:
    turns = [
        SimpleNamespace(
            prompt_ids=[1, 2],
            completion_ids=[3],
            completion_logprobs=[-0.1],
        ),
        SimpleNamespace(
            prompt_ids=[1, 2, 3, 50],
            completion_ids=[4],
            completion_logprobs=[-0.2],
            observation_mask=[0, 1],
        ),
    ]

    datum, stats = build_rollout_echo_datum(
        turns,
        completion_advantages=[[0.7], [0.4]],
        weight=0.05,
        min_prompt_overlap=1.0,
    )

    assert datum is not None
    assert datum.echo_advantages == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert datum.full_observation_count == 0
    assert datum.positive_tokens == 0
    assert stats.candidate_datums == 0
    assert stats.skipped_bad_observation_mask == 1


def test_limit_echo_masks_caps_without_truncating_rollout_rows() -> None:
    rows = [
        [0.0, 0.2, 0.2, 0.0],
        [0.1, 0.0, 0.1],
    ]

    limited, stats = limit_echo_masks(rows, max_positive_tokens=3)

    assert limited == [
        [0.0, 0.2, 0.2, 0.0],
        [0.1, 0.0, 0.0],
    ]
    assert [len(row) for row in limited] == [4, 3]
    assert stats.kept_datums == 2
    assert stats.kept_tokens == 3
    assert stats.kept_terminal_tokens == 0
    assert stats.truncated_tokens == 1


def test_limit_echo_masks_counts_only_terminal_tokens_retained_after_cap() -> None:
    rows = [
        [0.0, 0.2, 0.2],
        [0.1, 0.1],
    ]
    terminal_masks = [
        [0, 0, 0],
        [1, 1],
    ]

    limited, stats = limit_echo_masks(
        rows,
        max_positive_tokens=3,
        terminal_observation_masks=terminal_masks,
    )

    assert limited == [[0.0, 0.2, 0.2], [0.1, 0.0]]
    assert stats.kept_tokens == 3
    assert stats.kept_terminal_tokens == 1
    assert stats.truncated_tokens == 1


def test_retain_all_echo_masks_keeps_every_target_and_terminal_token() -> None:
    rows = [[0.0, 0.2, 0.2], [0.1, 0.0, 0.1]]
    terminal_masks = [[0, 0, 0], [1, 0, 1]]

    retained, stats = retain_all_echo_masks(
        rows,
        terminal_observation_masks=terminal_masks,
    )

    assert retained is rows
    assert stats == EchoLimitStats(
        kept_datums=2,
        kept_tokens=4,
        kept_terminal_tokens=2,
        truncated_tokens=0,
    )


def test_prepare_echo_step_plan_full_retention_bypasses_caps() -> None:
    acc = RolloutAccumulator(
        datum_echo_advantages=[[0.0, 0.2, 0.2], [0.2, 0.2]],
        datum_echo_terminal_masks=[[0, 0, 0], [1, 1]],
        echo_build=EchoBuildStats(candidate_tokens=4, terminal_candidate_tokens=2),
        sampled_completion_token_count=1,
        sampled_completion_surprisal_sum=0.0,
    )
    config = TrainConfig(
        echo_enabled=True,
        echo_target_retention="all",
        echo_entropy_floor=0.0,
        echo_max_tokens_per_step=1,
        echo_max_token_ratio=0.01,
    )

    plan = prepare_echo_step_plan(config, acc)

    assert plan.allowed_tokens == 4
    assert plan.limit == EchoLimitStats(
        kept_datums=2,
        kept_tokens=4,
        kept_terminal_tokens=2,
        truncated_tokens=0,
    )
    assert acc.datum_echo_advantages == [[0.0, 0.2, 0.2], [0.2, 0.2]]


def test_merge_echo_build_stats_adds_all_counters() -> None:
    left = EchoBuildStats(
        candidate_datums=1,
        candidate_tokens=2,
        observation_mask_datums=3,
        executed_transition_datums=4,
        skipped_first_turns=4,
        skipped_no_suffix=5,
        skipped_low_overlap=6,
        split_non_prefix=7,
        skipped_bad_observation_mask=8,
    )
    right = EchoBuildStats(
        candidate_datums=8,
        candidate_tokens=9,
        observation_mask_datums=10,
        executed_transition_datums=11,
        skipped_first_turns=11,
        skipped_no_suffix=12,
        skipped_low_overlap=13,
        split_non_prefix=14,
        skipped_bad_observation_mask=15,
    )

    assert merge_echo_build_stats(left, right) == EchoBuildStats(
        candidate_datums=9,
        candidate_tokens=11,
        observation_mask_datums=13,
        executed_transition_datums=15,
        skipped_first_turns=15,
        skipped_no_suffix=17,
        skipped_low_overlap=19,
        split_non_prefix=21,
        skipped_bad_observation_mask=23,
    )


def test_echo_treats_rl_advantages_as_opaque_algorithm_output() -> None:
    class Helper:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def train_step_with_echo_masks(
            self,
            all_tokens,
            all_logprobs,
            all_advantages,
            echo_advantages,
            echo_full_observation_counts,
            echo_loss_fn,
            lr,
            weight_decay,
            echo_rollout_denominator=0,
        ):
            self.calls.append(
                {
                    "tokens": all_tokens,
                    "logprobs": all_logprobs,
                    "advantages": all_advantages,
                    "echo_advantages": echo_advantages,
                    "echo_full_observation_counts": echo_full_observation_counts,
                    "echo_loss_fn": echo_loss_fn,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "echo_rollout_denominator": echo_rollout_denominator,
                }
            )
            return 0.25, 0.125

    helper = Helper()
    algorithm_advantages = [[0.7, -0.2, 0.4], [-0.5, -0.5]]

    rl_loss, echo_loss, joint = run_rl_echo_train_step(
        helper,
        all_tokens=[[10, 11, 12], [20, 21]],
        all_logprobs=[[-0.1, -0.2, -0.3], [-0.4, -0.5]],
        all_advantages=algorithm_advantages,
        echo_advantages=[
            [0.0, 0.0, 0.3],
            [0.0, 0.0],
        ],
        echo_full_observation_counts=[1, 0],
        echo_loss_fn="cross_entropy",
        lr=1e-4,
        weight_decay=0.01,
        echo_rollout_denominator=1,
    )

    assert (rl_loss, echo_loss, joint) == (0.25, 0.125, True)
    assert helper.calls[0]["advantages"] is algorithm_advantages
    assert helper.calls[0]["echo_advantages"] == [
        [0.0, 0.0, 0.3],
        [0.0, 0.0],
    ]
    assert helper.calls[0]["echo_full_observation_counts"] == [1, 0]
    assert helper.calls[0]["echo_rollout_denominator"] == 1


def test_echo_rejects_helper_without_shared_forward_step() -> None:
    class Helper:
        def train_step(self, *args, **kwargs):
            raise AssertionError("separate RL step must not run")

    with pytest.raises(RuntimeError, match="same rollout rows"):
        run_rl_echo_train_step(
            Helper(),
            all_tokens=[[10, 11]],
            all_logprobs=[[-0.1, -0.2]],
            all_advantages=[[0.5, -0.5]],
            echo_advantages=[[0.0, 0.0, 0.3]],
            echo_full_observation_counts=[1],
            echo_loss_fn="cross_entropy",
            lr=1e-4,
            weight_decay=0.0,
        )
