import pytest

from retrain.backends.unsloth import validate_fast_language_model_api


def test_installed_unsloth_fast_language_model_api_is_supported():
    unsloth = pytest.importorskip("unsloth")

    payload = validate_fast_language_model_api(unsloth.FastLanguageModel)

    assert "from_pretrained_params" in payload
    assert "get_peft_model_params" in payload
