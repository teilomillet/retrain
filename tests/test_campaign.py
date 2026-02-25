"""Tests for retrain.campaign condition parsing and validation."""

import pytest

from retrain.campaign import DEFAULT_CONDITIONS, _parse_campaign_conditions


class TestParseCampaignConditions:
    def test_defaults_when_conditions_missing(self):
        assert _parse_campaign_conditions(None, "campaign.toml") == list(
            DEFAULT_CONDITIONS
        )
        assert _parse_campaign_conditions([], "campaign.toml") == list(
            DEFAULT_CONDITIONS
        )

    def test_valid_conditions_with_new_modes(self):
        conditions = _parse_campaign_conditions(
            [
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "gtpo_sepa_amp",
                },
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "gtpo_sepa_amp_c",
                },
            ],
            "campaign.toml",
        )
        assert conditions == [
            ("maxrl", "gtpo_sepa_amp"),
            ("maxrl", "gtpo_sepa_amp_c"),
        ]

    def test_dotted_transform_mode_is_accepted(self):
        conditions = _parse_campaign_conditions(
            [
                {
                    "advantage_mode": "maxrl",
                    "transform_mode": "my_transforms.make_transform_spec",
                }
            ],
            "campaign.toml",
        )
        assert conditions == [("maxrl", "my_transforms.make_transform_spec")]

    def test_non_list_conditions_raises(self):
        with pytest.raises(ValueError, match="campaign.conditions must be a list"):
            _parse_campaign_conditions({"advantage_mode": "grpo"}, "campaign.toml")

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="advantage_mode must be a non-empty string"):
            _parse_campaign_conditions([{"transform_mode": "none"}], "campaign.toml")
        with pytest.raises(ValueError, match="transform_mode must be a non-empty string"):
            _parse_campaign_conditions([{"advantage_mode": "grpo"}], "campaign.toml")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid campaign condition at index 0"):
            _parse_campaign_conditions(
                [{"advantage_mode": "grpo", "transform_mode": "typo"}],
                "campaign.toml",
            )
