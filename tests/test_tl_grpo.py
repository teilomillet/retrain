"""Unit tests for TL-GRPO (Turn-Level Group Relative Policy Optimization)."""

from __future__ import annotations

import pytest

from retrain.verifiers_bridge import (
    _compute_tl_grpo_advantages,
    _run_tl_grpo_branching,
    VerifiersTurnSample,
)


# ---------------------------------------------------------------------------
# _compute_tl_grpo_advantages tests
# ---------------------------------------------------------------------------


class TestComputeTlGrpoAdvantages:
    """Pure-math advantage computation from pre-collected branch rewards."""

    def test_same_rewards_give_zero_local_advantage(self):
        """When all branches get the same reward, local advantage is 0."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0, "reward": 0.5},
        ]
        # 3 turns, 4 branches each, all identical rewards
        branch_rewards = [
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.5)
        advs = states[0]["turn_advantages"]
        assert len(advs) == 3
        # With zero outcome advantage and identical branch rewards,
        # all turn advantages should be 0 (or very close).
        for a in advs:
            assert abs(a) < 1e-6

    def test_better_action_gets_positive_advantage(self):
        """Primary action scoring higher than alternatives → positive local."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0},
        ]
        # Turn 0: primary=10, alternatives=0,0,0 → primary is best
        branch_rewards = [
            [[10.0, 0.0, 0.0, 0.0]],
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.0)
        advs = states[0]["turn_advantages"]
        assert len(advs) == 1
        assert advs[0] > 0.0

    def test_worse_action_gets_negative_advantage(self):
        """Primary action scoring lower than alternatives → negative local."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0},
        ]
        # Turn 0: primary=0, alternatives=10,10,10 → primary is worst
        branch_rewards = [
            [[0.0, 10.0, 10.0, 10.0]],
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.0)
        advs = states[0]["turn_advantages"]
        assert len(advs) == 1
        assert advs[0] < 0.0

    def test_outcome_advantage_blended(self):
        """Episode-level advantage is blended via turn_weight."""
        states: list[dict[str, object]] = [
            {"advantage": 2.0},
        ]
        # Same branch rewards → local advantage ~0, but outcome contributes.
        branch_rewards = [
            [[5.0, 5.0, 5.0, 5.0]],
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.5)
        advs = states[0]["turn_advantages"]
        assert len(advs) == 1
        # Should be ~0.0 (local) + 0.5 * 2.0 (outcome) = 1.0
        assert abs(advs[0] - 1.0) < 1e-6

    def test_single_branch_falls_back_to_outcome(self):
        """With only 1 branch (no alternatives), advantage = turn_weight * outcome."""
        states: list[dict[str, object]] = [
            {"advantage": 3.0},
        ]
        branch_rewards = [
            [[5.0]],  # Only primary, no alternatives
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.5)
        advs = states[0]["turn_advantages"]
        assert len(advs) == 1
        assert abs(advs[0] - 1.5) < 1e-6  # 0.5 * 3.0

    def test_multiple_rollouts(self):
        """Multiple rollouts each get independent advantages."""
        states: list[dict[str, object]] = [
            {"advantage": 1.0},
            {"advantage": -1.0},
        ]
        branch_rewards = [
            [[10.0, 0.0]],  # rollout 0: primary is best
            [[0.0, 10.0]],  # rollout 1: primary is worst
        ]
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.0)
        assert states[0]["turn_advantages"][0] > 0.0
        assert states[1]["turn_advantages"][0] < 0.0

    def test_empty_branch_rewards(self):
        """State with no branch rewards → empty turn_advantages."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0},
        ]
        _compute_tl_grpo_advantages(states, [[]])
        assert states[0]["turn_advantages"] == []

    def test_missing_rollout_in_branch_rewards(self):
        """If branch_rewards has fewer entries than states, extras get []."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0, "reward": 0.5},
            {"advantage": 0.0, "reward": 0.5},
        ]
        branch_rewards = [
            [[5.0, 5.0]],
        ]
        _compute_tl_grpo_advantages(states, branch_rewards)
        assert len(states[0]["turn_advantages"]) == 1
        assert states[1]["turn_advantages"] == []

    def test_outcome_baseline_overrides_group_advantage(self):
        """With outcome_baseline, uses R - baseline instead of state advantage."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0, "reward": 0.85},  # group_size=1 → advantage=0
        ]
        # All branches identical → local advantage = 0
        branch_rewards = [
            [[0.0, 0.0, 0.0, 0.0]],
        ]
        # Without baseline: advantage=0, turn_adv = 0 + 0.5*0 = 0
        _compute_tl_grpo_advantages(states, branch_rewards, turn_weight=0.5)
        assert abs(states[0]["turn_advantages"][0]) < 1e-6

        # With baseline=0.5: outcome = 0.85 - 0.5 = 0.35, turn_adv = 0 + 0.5*0.35 = 0.175
        _compute_tl_grpo_advantages(
            states, branch_rewards, turn_weight=0.5, outcome_baseline=0.5,
        )
        assert abs(states[0]["turn_advantages"][0] - 0.175) < 1e-6

    def test_outcome_baseline_bad_episode(self):
        """Bad episode with baseline → negative outcome signal."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0, "reward": 0.20},
        ]
        branch_rewards = [
            [[0.0, 0.0, 0.0, 0.0]],
        ]
        _compute_tl_grpo_advantages(
            states, branch_rewards, turn_weight=0.5, outcome_baseline=0.5,
        )
        # outcome = 0.20 - 0.5 = -0.30, turn_adv = 0 + 0.5 * (-0.30) = -0.15
        assert abs(states[0]["turn_advantages"][0] - (-0.15)) < 1e-6

    def test_outcome_baseline_combines_with_local(self):
        """Baseline outcome + nonzero local advantage combine correctly."""
        states: list[dict[str, object]] = [
            {"advantage": 0.0, "reward": 0.80},
        ]
        # Primary is best → positive local advantage
        branch_rewards = [
            [[10.0, 0.0, 0.0, 0.0]],
        ]
        _compute_tl_grpo_advantages(
            states, branch_rewards, turn_weight=0.5, outcome_baseline=0.5,
        )
        adv = states[0]["turn_advantages"][0]
        # local > 0 (primary is best) + 0.5 * (0.80 - 0.5) = local + 0.15
        assert adv > 0.15  # local is positive, so total > 0.15


# ---------------------------------------------------------------------------
# _run_tl_grpo_branching tests (with mocks)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stub that tracks execute() calls and returns canned responses."""

    def __init__(self, cumulative_rewards: dict[int, float] | None = None):
        self._cumulative_rewards = cumulative_rewards or {}
        self.calls: list[list[object]] = []

    def execute(self, operations: list[object], **kwargs: object) -> dict[str, object]:
        self.calls.append(list(operations))
        n_ops = len(operations)
        cum = self._cumulative_rewards.get(n_ops, 0.0)
        return {
            "run": {"cumulative_reward": cum},
            "evaluation": {"completed": False, "passed": False, "score": 0.0},
            "model_view": {},
        }


class _FakeDomain:
    def extract_operation(self, text: str) -> dict[str, object]:
        if "bad" in text:
            raise ValueError("bad action")
        return {"kind": "wait"}


class _FakeHelper:
    """Stub helper.sample() that returns canned completions."""

    def __init__(self, texts: list[str]):
        self._texts = texts

    def sample(
        self,
        prompt_ids_batch: list[list[int]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[tuple[list[int], list[float]]]]:
        # Return n samples per prompt. Token ids are dummy.
        result: list[list[tuple[list[int], list[float]]]] = []
        for _prompt in prompt_ids_batch:
            samples = []
            for i in range(n):
                idx = i % len(self._texts)
                text = self._texts[idx]
                ids = list(range(len(text)))
                logprobs = [0.0] * len(ids)
                samples.append((ids, logprobs))
            result.append(samples)
        return result


class _FakeTokenizer:
    def __init__(self, decode_map: dict[int, str] | None = None):
        self._decode_map = decode_map or {}

    def batch_decode(
        self, token_ids_batch: list[list[int]], *, skip_special_tokens: bool = True
    ) -> list[str]:
        results = []
        for ids in token_ids_batch:
            n = len(ids)
            results.append(self._decode_map.get(n, '{"kind":"wait"}'))
        return results


class _FakeEnv:
    """Minimal env with domain for testing branching."""

    def __init__(self) -> None:
        self.domain = _FakeDomain()


class TestRunTlGrpoBranching:
    def _make_state(
        self,
        turn_log: list[dict[str, object]],
        client: _FakeClient,
    ) -> dict[str, object]:
        env = type("E", (), {"client": client, "fork_execute": client.execute})()
        return {
            "env": env,
            "turn_log": turn_log,
        }

    def test_basic_branching(self):
        """Branching from a single valid turn produces G reward deltas."""
        # Primary turn: delta=5.0, cumulative=5.0
        turn_log = [
            {
                "turn": 1,
                "operation": {"kind": "wait"},
                "reward_delta": 5.0,
                "cumulative_reward": 5.0,
                "valid": True,
            },
        ]
        # Alternative actions will execute 1 op (ops_before=[] + alt_op).
        # Client returns cumulative=3.0 for 1 op → delta = 3.0 - 0.0 = 3.0
        client = _FakeClient(cumulative_rewards={1: 3.0})
        state = self._make_state(turn_log, client)

        env = _FakeEnv()
        helper = _FakeHelper(texts=['{"kind":"wait"}'])
        tokenizer = _FakeTokenizer()
        turns = [
            VerifiersTurnSample(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5],
                completion_logprobs=[0.0, 0.0],
                completion_text='{"kind":"wait"}',
            ),
        ]

        result = _run_tl_grpo_branching(
            state, turns, env, helper, tokenizer,
            branch_size=4, max_tokens=100, temperature=0.7, top_p=0.95,
        )
        assert len(result) == 1
        assert len(result[0]) == 4  # primary + 3 alternatives
        assert result[0][0] == 5.0  # primary delta

    def test_invalid_turn_not_branched(self):
        """Invalid turns get a single-element reward list (no branching)."""
        turn_log = [
            {
                "turn": 1,
                "operation": None,
                "reward_delta": 0.0,
                "cumulative_reward": 0.0,
                "valid": False,
            },
        ]
        client = _FakeClient()
        state = self._make_state(turn_log, client)
        env = _FakeEnv()
        helper = _FakeHelper(texts=['{"kind":"wait"}'])
        tokenizer = _FakeTokenizer()
        turns = [
            VerifiersTurnSample(
                prompt_ids=[1], completion_ids=[2],
                completion_logprobs=[0.0], completion_text="bad",
            ),
        ]

        result = _run_tl_grpo_branching(
            state, turns, env, helper, tokenizer, branch_size=4,
        )
        assert len(result) == 1
        assert result[0] == [0.0]
        assert len(client.calls) == 0  # no kernel calls for invalid turns

    def test_bad_alternative_gets_zero_delta(self):
        """If an alternative can't be parsed, its delta is 0.0."""
        turn_log = [
            {
                "turn": 1,
                "operation": {"kind": "wait"},
                "reward_delta": 2.0,
                "cumulative_reward": 2.0,
                "valid": True,
            },
        ]
        client = _FakeClient(cumulative_rewards={1: 1.0})
        state = self._make_state(turn_log, client)

        env = _FakeEnv()
        # "bad" text triggers ValueError in extract_operation
        helper = _FakeHelper(texts=["bad action"])
        tokenizer = _FakeTokenizer(decode_map={10: "bad action"})

        turns = [
            VerifiersTurnSample(
                prompt_ids=[1, 2], completion_ids=[3],
                completion_logprobs=[0.0], completion_text='{"kind":"wait"}',
            ),
        ]

        result = _run_tl_grpo_branching(
            state, turns, env, helper, tokenizer,
            branch_size=3, max_tokens=100, temperature=0.7, top_p=0.95,
        )
        assert len(result) == 1
        assert result[0][0] == 2.0  # primary
        assert result[0][1] == 0.0  # bad alternative
        assert result[0][2] == 0.0  # bad alternative

    def test_no_env_client_returns_empty(self):
        """If state has no env client, branching returns []."""
        state: dict[str, object] = {
            "env": None,
            "turn_log": [{"turn": 1, "valid": True, "operation": {"kind": "wait"},
                          "reward_delta": 1.0, "cumulative_reward": 1.0}],
        }
        result = _run_tl_grpo_branching(
            state, [], _FakeEnv(), _FakeHelper([""]), _FakeTokenizer(),
        )
        assert result == []

    def test_ops_before_accumulates(self):
        """Each subsequent turn includes prior valid operations in replay."""
        turn_log = [
            {
                "turn": 1,
                "operation": {"kind": "wait"},
                "reward_delta": 1.0,
                "cumulative_reward": 1.0,
                "valid": True,
            },
            {
                "turn": 2,
                "operation": {"kind": "act", "action": {"type": "accept_customer"}},
                "reward_delta": 3.0,
                "cumulative_reward": 4.0,
                "valid": True,
            },
        ]
        # For turn 0: ops_before = [], alt executes 1 op → cum_rewards[1]
        # For turn 1: ops_before = [wait], alt executes 2 ops → cum_rewards[2]
        client = _FakeClient(cumulative_rewards={1: 0.5, 2: 2.0})
        state = self._make_state(turn_log, client)
        env = _FakeEnv()
        helper = _FakeHelper(texts=['{"kind":"wait"}'])
        tokenizer = _FakeTokenizer()
        turns = [
            VerifiersTurnSample(
                prompt_ids=[1], completion_ids=[2],
                completion_logprobs=[0.0], completion_text='{"kind":"wait"}',
            ),
            VerifiersTurnSample(
                prompt_ids=[1, 2, 3], completion_ids=[4],
                completion_logprobs=[0.0], completion_text='{"kind":"wait"}',
            ),
        ]

        result = _run_tl_grpo_branching(
            state, turns, env, helper, tokenizer,
            branch_size=2, max_tokens=100, temperature=0.7, top_p=0.95,
        )
        assert len(result) == 2
        # Turn 0: primary=1.0, alt=0.5 (cum[1] - pre_cum=0.0)
        assert result[0][0] == 1.0
        assert result[0][1] == 0.5
        # Turn 1: primary=3.0, alt delta = cum[2] - pre_cum = 2.0 - 1.0 = 1.0
        assert result[1][0] == 3.0
        assert result[1][1] == pytest.approx(1.0)
