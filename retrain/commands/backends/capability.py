"""Backend capability payloads for CLI display."""

from __future__ import annotations


def payload(
    backend_name: str,
    backend_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Resolve a backend's capabilities into a JSON-friendly mapping."""
    from retrain.backends.catalog import (
        backend_capability_source,
        resolve_backend_capabilities,
    )

    options = backend_options or {}
    caps = resolve_backend_capabilities(backend_name, options)
    return {
        "backend": backend_name,
        "source": backend_capability_source(backend_name, options),
        "reports_sync_loss": caps.reports_sync_loss,
        "preserves_token_advantages": caps.preserves_token_advantages,
        "supports_checkpoint_resume": caps.supports_checkpoint_resume,
        "resume_runtime_dependent": caps.resume_runtime_dependent,
        "supports_echo_shared_forward": caps.supports_echo_shared_forward,
    }


def summary(capabilities: dict[str, object]) -> str:
    """One-line rendering of a capability payload."""
    return (
        f"source={capabilities['source']}, "
        f"reports_sync_loss={capabilities['reports_sync_loss']}, "
        f"preserves_token_advantages={capabilities['preserves_token_advantages']}, "
        f"supports_checkpoint_resume={capabilities['supports_checkpoint_resume']}, "
        f"resume_runtime_dependent={capabilities['resume_runtime_dependent']}, "
        f"supports_echo_shared_forward={capabilities['supports_echo_shared_forward']}"
    )
