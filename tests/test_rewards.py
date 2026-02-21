"""Tests for retrain.rewards — extract_boxed, BoxedMathReward, and reward factory."""

import sys
import types
from unittest import mock

import pytest

from retrain.config import TrainConfig
from retrain.rewards import (
    BoxedMathReward,
    CustomReward,
    RewardFunction,
    create_reward,
    extract_boxed,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_deeply_nested(self):
        assert extract_boxed(r"\boxed{a{b{c}d}e}") == "a{b{c}d}e"

    def test_multiple_takes_last(self):
        assert extract_boxed(r"\boxed{wrong} then \boxed{right}") == "right"

    def test_no_boxed(self):
        assert extract_boxed("no boxed here") == ""

    def test_empty_boxed(self):
        assert extract_boxed(r"\boxed{}") == ""

    def test_unclosed_brace(self):
        # Unclosed brace — should still extract what's inside
        result = extract_boxed(r"\boxed{abc")
        assert result == "abc"

    def test_whitespace_stripping(self):
        assert extract_boxed(r"\boxed{  hello  }") == "hello"

    def test_empty_string(self):
        assert extract_boxed("") == ""

    def test_boxed_at_end_no_content(self):
        assert extract_boxed(r"\boxed{") == ""


class TestBoxedMathReward:
    def setup_method(self):
        self.reward = BoxedMathReward()

    def test_correct(self):
        assert self.reward.score(r"Answer is \boxed{42}", "42") == 1.0

    def test_incorrect(self):
        assert self.reward.score(r"Answer is \boxed{41}", "42") == 0.0

    def test_no_boxed(self):
        assert self.reward.score("no answer", "42") == 0.0

    def test_whitespace_match(self):
        assert self.reward.score(r"\boxed{ 42 }", " 42 ") == 1.0

    def test_latex_expression(self):
        ref = r"\frac{1}{2}"
        assert self.reward.score(rf"\boxed{{{ref}}}", ref) == 1.0

    def test_implements_protocol(self):
        assert isinstance(BoxedMathReward(), RewardFunction)


class TestCreateRewardFactory:
    def test_default_returns_boxed(self):
        config = TrainConfig()
        reward = create_reward(config)
        assert isinstance(reward, BoxedMathReward)

    def test_match_type_returns_boxed(self):
        config = TrainConfig(reward_type="match")
        reward = create_reward(config)
        assert isinstance(reward, BoxedMathReward)

    def test_unknown_type_raises(self):
        config = TrainConfig(reward_type="nonexistent")
        with pytest.raises(ValueError, match="Unknown reward type 'nonexistent'"):
            create_reward(config)

    def test_math_type_without_verifiers_raises(self):
        config = TrainConfig(reward_type="math")
        with mock.patch.dict(sys.modules, {"verifiers": None}):
            with pytest.raises(ImportError, match="pip install retrain\\[verifiers\\]"):
                create_reward(config)

    def test_judge_type_without_verifiers_raises(self):
        config = TrainConfig(reward_type="judge")
        with mock.patch.dict(sys.modules, {"verifiers": None}):
            with pytest.raises(ImportError, match="pip install retrain\\[verifiers\\]"):
                create_reward(config)

    def test_custom_type_missing_module_raises(self):
        config = TrainConfig(reward_type="custom", reward_custom_module="")
        with pytest.raises(ValueError, match="custom_module"):
            create_reward(config)


class TestCustomReward:
    def test_sync_function(self):
        # Create a fake module with a sync score function
        mod = types.ModuleType("fake_reward_mod")
        mod.my_score = lambda response, reference: 1.0 if "correct" in response else 0.0
        with mock.patch.dict(sys.modules, {"fake_reward_mod": mod}):
            reward = CustomReward("fake_reward_mod", "my_score")
            assert reward.score("this is correct", "ref") == 1.0
            assert reward.score("this is wrong", "ref") == 0.0

    def test_async_function(self):
        mod = types.ModuleType("fake_async_mod")

        async def async_score(response, reference):
            return 0.75

        mod.score = async_score
        with mock.patch.dict(sys.modules, {"fake_async_mod": mod}):
            reward = CustomReward("fake_async_mod", "score")
            assert reward.score("anything", "ref") == 0.75

    def test_missing_function_raises(self):
        mod = types.ModuleType("fake_empty_mod")
        with mock.patch.dict(sys.modules, {"fake_empty_mod": mod}):
            with pytest.raises(AttributeError, match="no attribute 'missing_fn'"):
                CustomReward("fake_empty_mod", "missing_fn")

    def test_custom_via_factory(self):
        mod = types.ModuleType("factory_test_mod")
        mod.score = lambda r, ref: 0.5
        with mock.patch.dict(sys.modules, {"factory_test_mod": mod}):
            config = TrainConfig(
                reward_type="custom",
                reward_custom_module="factory_test_mod",
                reward_custom_function="score",
            )
            reward = create_reward(config)
            assert isinstance(reward, CustomReward)
            assert reward.score("x", "y") == 0.5

    def test_implements_protocol(self):
        mod = types.ModuleType("proto_mod")
        mod.score = lambda r, ref: 0.0
        with mock.patch.dict(sys.modules, {"proto_mod": mod}):
            assert isinstance(CustomReward("proto_mod", "score"), RewardFunction)
