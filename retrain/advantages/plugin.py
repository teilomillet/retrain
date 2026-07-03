"""Shared plugin registry coercion for advantage modules."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from typing import TypeVar

from retrain.advantages.types import (
    AdvantageContext,
    AdvantageOutput,
    AdvantageResult,
    AdvantageSpec,
    AlgorithmOutput,
    EntropyStats,
    EpisodeAdvantageComputeFn,
    EpisodeAdvantageRunner,
    TransformOutput,
)

SpecT = TypeVar("SpecT")


def _is_dotted_plugin_path(name: str) -> bool:
    module_path, _, attr_name = name.rpartition(".")
    return bool(module_path and attr_name)


def _validate_short_registry_name(name: str, *, label: str) -> None:
    if not name or "." in name:
        raise ValueError(
            f"{label} name must be non-empty and cannot contain '.'. "
            "Use dotted paths directly in TOML for external plugins."
        )


def _resolve_registry_spec(
    name: str,
    *,
    builtins: Mapping[str, SpecT],
    cache: dict[str, SpecT],
    load_custom: Callable[[str], SpecT],
    builtin_names: Callable[[], list[str]],
    config_key: str,
    custom_label: str,
    example: str,
) -> SpecT:
    cached = cache.get(name)
    if cached is not None:
        return cached

    spec = builtins.get(name)
    if spec is None:
        if _is_dotted_plugin_path(name):
            spec = load_custom(name)
        else:
            raise ValueError(
                f"Unknown {config_key} '{name}'. "
                f"Built-in options: {builtin_names()}. "
                f"For custom {custom_label} use dotted path format "
                f"(e.g. '{example}')."
            )
    cache[name] = spec
    return spec


# ---------------------------------------------------------------------------
def _normalize_advantage_compute(
    compute: EpisodeAdvantageComputeFn,
) -> EpisodeAdvantageRunner:
    """Adapt a plugin function to the internal `(rewards, params)` signature."""
    try:
        sig = inspect.signature(compute)
    except (TypeError, ValueError):
        return lambda rewards, _params: compute(rewards)  # type: ignore[missing-argument]

    params = sig.parameters
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1 and positional[0].name in {"ctx", "context"}:
        def _run_ctx_style(
            rewards: list[float], params_map: Mapping[str, object]
        ) -> list[float]:
            out = compute(AdvantageContext(rewards=rewards, params=params_map))  # type: ignore[missing-argument, invalid-argument-type]
            if isinstance(out, AdvantageOutput):
                return out.advantages
            return out

        return _run_ctx_style

    params_arg = params.get("params")
    accepts_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if params_arg is not None and params_arg.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        if params_arg.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            return lambda rewards, params_map: compute(rewards, params_map)  # type: ignore[too-many-positional-arguments]
        return lambda rewards, params_map: compute(rewards, params=params_map)  # type: ignore[missing-argument, unknown-argument]

    if accepts_varkw:
        return lambda rewards, params_map: compute(rewards, params=params_map)  # type: ignore[missing-argument, unknown-argument]

    return lambda rewards, _params: compute(rewards)  # type: ignore[missing-argument]


def _coerce_advantages_output(
    raw_advantages: object, expected_len: int, mode_name: str
) -> list[float]:
    """Normalize and validate episode-level advantage output shape."""
    try:
        advantages = [float(v) for v in raw_advantages]  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"advantage_mode '{mode_name}' must return an iterable of floats."
        ) from exc

    if len(advantages) != expected_len:
        raise ValueError(
            f"advantage_mode '{mode_name}' returned {len(advantages)} values, "
            f"expected {expected_len}."
        )
    for idx, value in enumerate(advantages):
        if not math.isfinite(value):
            raise ValueError(
                f"advantage_mode '{mode_name}' returned non-finite value "
                f"at index {idx}: {value!r}"
            )
    return advantages


def _as_advantage_spec(
    name: str, compute: EpisodeAdvantageComputeFn
) -> AdvantageSpec:
    """Build an AdvantageSpec from a one-arg or two-arg callable."""
    return AdvantageSpec(name=name, compute=_normalize_advantage_compute(compute))


def _callable_takes_no_positional_args(obj: object) -> bool:
    """Return True when callable is likely a zero-arg factory."""
    if not callable(obj):
        return False
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
    )
    return not positional and not accepts_varargs


def _validate_token_advs(
    token_advs: list[list[float]],
    *,
    expected_lens: list[int],
    mode_name: str,
) -> None:
    """Validate shape and numeric sanity for token-level advantages."""
    if len(token_advs) != len(expected_lens):
        raise ValueError(
            f"{mode_name} returned {len(token_advs)} sequences, "
            f"expected {len(expected_lens)}."
        )
    for seq_idx, (seq, expected_len) in enumerate(zip(token_advs, expected_lens)):
        if len(seq) != expected_len:
            raise ValueError(
                f"{mode_name} returned {len(seq)} token advantages for sequence "
                f"{seq_idx}, expected {expected_len}."
            )
        for tok_idx, value in enumerate(seq):
            if not math.isfinite(value):
                raise ValueError(
                    f"{mode_name} returned non-finite value at sequence {seq_idx}, "
                    f"token {tok_idx}: {value!r}"
                )


def _coerce_transform_output(
    raw_output: object,
    *,
    expected_lens: list[int],
    mode_name: str,
) -> AdvantageResult:
    """Normalize transform plugin output to AdvantageResult."""
    if isinstance(raw_output, AdvantageResult):
        result = raw_output
    elif isinstance(raw_output, TransformOutput):
        result = AdvantageResult(
            token_advs=raw_output.token_advs,
            has_stats=raw_output.has_stats,
            stats=raw_output.stats,
            extra_metrics=raw_output.extra_metrics,
        )
    else:
        token_advs = raw_output
        result = AdvantageResult(
            token_advs=[[float(v) for v in seq] for seq in token_advs],  # type: ignore[arg-type]
            has_stats=False,
            stats=EntropyStats(),
            extra_metrics={},
        )

    _validate_token_advs(
        result.token_advs,
        expected_lens=expected_lens,
        mode_name=mode_name,
    )
    return result


def _coerce_algorithm_output(
    raw_output: object,
    *,
    expected_lens: list[int],
    mode_name: str,
) -> AdvantageResult:
    """Normalize algorithm plugin output to AdvantageResult."""
    if isinstance(raw_output, AlgorithmOutput):
        result = AdvantageResult(
            token_advs=raw_output.token_advs,
            has_stats=raw_output.has_stats,
            stats=raw_output.stats,
            extra_metrics=raw_output.extra_metrics,
        )
    else:
        result = _coerce_transform_output(
            raw_output,
            expected_lens=expected_lens,
            mode_name=mode_name,
        )
        return result

    _validate_token_advs(
        result.token_advs,
        expected_lens=expected_lens,
        mode_name=mode_name,
    )
    return result


# ---------------------------------------------------------------------------
