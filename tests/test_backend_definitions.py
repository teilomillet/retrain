"""Tests for backend capability metadata resolution."""

import sys
from types import SimpleNamespace

from retrain.backend_definitions import (
    BackendOptionSpec,
    backend_capability_source,
    describe_backends_catalog,
    normalize_backend_options,
    resolve_backend_capabilities,
)


def _as_tuple(backend_name: str) -> tuple[bool, bool, bool, bool]:
    caps = resolve_backend_capabilities(backend_name, {})
    return (
        caps.reports_sync_loss,
        caps.preserves_token_advantages,
        caps.supports_checkpoint_resume,
        caps.resume_runtime_dependent,
    )


def test_builtin_capabilities_match_spec() -> None:
    assert _as_tuple("local") == (True, True, True, False)
    assert _as_tuple("tinker") == (True, True, True, True)
    assert _as_tuple("prime_rl") == (False, False, True, False)
    assert backend_capability_source("local") == "builtin"


def test_plugin_capabilities_use_conservative_defaults() -> None:
    assert _as_tuple("my_backend.Factory") == (True, False, True, False)
    assert backend_capability_source("my_backend.Factory") == "plugin/default"


def test_resolver_accepts_backend_options_for_builtins() -> None:
    caps = resolve_backend_capabilities(
        "prime_rl",
        {
            "transport": "zmq",
            "strict_advantages": True,
        },
    )
    assert caps.reports_sync_loss is False
    assert caps.preserves_token_advantages is False


def test_local_backend_options_normalize_bool_values() -> None:
    normalized = normalize_backend_options(
        "local",
        {
            "trust_remote_code": "true",
            "require_causal_conv1d": "false",
            "train_microbatch_size": "1",
            "cuda_empty_cache": "true",
            "sample_use_cache": "false",
        },
    )
    assert normalized == {
        "trust_remote_code": True,
        "require_causal_conv1d": False,
        "train_microbatch_size": 1,
        "cuda_empty_cache": True,
        "sample_use_cache": False,
    }


def test_local_backend_options_reject_negative_microbatch() -> None:
    try:
        normalize_backend_options("local", {"train_microbatch_size": -1})
    except ValueError as exc:
        assert "train_microbatch_size" in str(exc)
        assert "must be >= 0" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plugin_capability_hook_and_option_schema(monkeypatch) -> None:
    class _PluginFactory:
        retrain_backend_capabilities = {
            "reports_sync_loss": False,
            "preserves_token_advantages": True,
            "supports_checkpoint_resume": True,
            "resume_runtime_dependent": True,
        }
        retrain_backend_option_schema = {
            "workers": BackendOptionSpec(value_type=int, default=2),
            "mode": {
                "type": "str",
                "default": "fast",
                "choices": ("fast", "safe"),
            },
        }

    monkeypatch.setitem(
        sys.modules,
        "plugin_mod",
        SimpleNamespace(PluginFactory=_PluginFactory),
    )

    caps = resolve_backend_capabilities(
        "plugin_mod.PluginFactory",
        {"workers": "4"},
    )
    assert caps.reports_sync_loss is False
    assert caps.resume_runtime_dependent is True
    assert (
        backend_capability_source("plugin_mod.PluginFactory", {"workers": "4"})
        == "plugin/hook"
    )

    normalized = normalize_backend_options(
        "plugin_mod.PluginFactory",
        {"workers": "4", "mode": "safe"},
    )
    assert normalized == {"workers": 4, "mode": "safe"}


def test_backends_catalog_payload_shape() -> None:
    payload = describe_backends_catalog()
    names = {item["name"] for item in payload["builtins"]}
    assert {"local", "tinker", "prime_rl"} <= names
    assert "capability_hooks" in payload["plugin"]
