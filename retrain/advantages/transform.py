"""Transform modes and registry."""

from __future__ import annotations

from retrain.advantages.credit import (
    apply_sepa_amplification,
    apply_sepa_amplification_clamped,
    apply_sepa_pooling,
    surprisal_mask_post_process,
)
from retrain.advantages.delight import (
    _compute_delight_sepa_transform,
    _compute_delight_transform,
    _compute_hard_delight_transform,
)
from retrain.advantages.plugin import (
    _callable_takes_no_positional_args,
    _is_dotted_plugin_path,
    _resolve_registry_spec,
    _validate_short_registry_name,
)
from retrain.advantages.types import TransformContextFn, TransformSpec
from retrain.plugins.resolve import resolve_dotted_attribute

_BUILTIN_TRANSFORM_SPECS: dict[str, TransformSpec] = {
    "none": TransformSpec(name="none", use_gtpo=False),
    "delight": TransformSpec(
        name="delight",
        use_gtpo=False,
        compute_context=_compute_delight_transform,
    ),
    "delight_sepa": TransformSpec(
        name="delight_sepa",
        use_gtpo=False,
        uses_sepa_controller=True,
        compute_context=_compute_delight_sepa_transform,
    ),
    "hard_delight": TransformSpec(
        name="hard_delight",
        use_gtpo=False,
        uses_sepa_controller=True,
        compute_context=_compute_hard_delight_transform,
    ),
    "gtpo": TransformSpec(name="gtpo"),
    "entropy_mask": TransformSpec(name="entropy_mask", use_gtpo=True, post_process=surprisal_mask_post_process),
    "gtpo_hicra": TransformSpec(
        name="gtpo_hicra", needs_planning=True, apply_hicra=True
    ),
    "gtpo_sepa": TransformSpec(
        name="gtpo_sepa",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_pooling,
    ),
    "gtpo_sepa_amp": TransformSpec(
        name="gtpo_sepa_amp",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_amplification,
    ),
    "gtpo_sepa_amp_c": TransformSpec(
        name="gtpo_sepa_amp_c",
        needs_planning=True,
        uses_sepa_controller=True,
        entropy_transform=apply_sepa_amplification_clamped,
    ),
}

_TRANSFORM_SPEC_CACHE: dict[str, TransformSpec] = {}


def register_transform_mode(name: str, spec_or_fn: TransformSpec | TransformContextFn) -> None:
    """Register or replace a short-name transform mode at runtime."""
    _validate_short_registry_name(name, label="Transform mode")
    if isinstance(spec_or_fn, TransformSpec):
        spec = spec_or_fn
    elif callable(spec_or_fn):
        spec = TransformSpec(name=name, compute_context=spec_or_fn)
    else:
        raise TypeError(
            f"Transform registration for '{name}' must be TransformSpec "
            f"or callable, got {type(spec_or_fn).__name__}."
        )
    _BUILTIN_TRANSFORM_SPECS[name] = spec
    _TRANSFORM_SPEC_CACHE.pop(name, None)


def get_builtin_transform_modes() -> list[str]:
    """Return sorted built-in transform mode names."""
    return sorted(_BUILTIN_TRANSFORM_SPECS)


def is_valid_transform_mode_name(transform_mode: str) -> bool:
    """True for built-ins or dotted plugin paths (`module.attr`)."""
    if transform_mode in _BUILTIN_TRANSFORM_SPECS:
        return True
    return _is_dotted_plugin_path(transform_mode)


def _load_custom_transform_spec(dotted_path: str) -> TransformSpec:
    resolved = resolve_dotted_attribute(
        dotted_path,
        selector="transform_mode",
        expected="TransformSpec, context callable, or zero-arg factory",
    )
    obj = resolved.obj

    if isinstance(obj, TransformSpec):
        return obj
    if callable(obj):
        if _callable_takes_no_positional_args(obj):
            built = obj()  # type: ignore[call-top-callable]
            if isinstance(built, TransformSpec):
                return built
            if callable(built):
                return TransformSpec(
                    name=dotted_path,
                    needs_planning=bool(getattr(built, "needs_planning", False)),
                    uses_sepa_controller=bool(
                        getattr(built, "uses_sepa_controller", False)
                    ),
                    compute_context=built,  # type: ignore[invalid-argument-type]
                )
            raise TypeError(
                f"transform_mode '{dotted_path}' factory returned "
                f"{type(built).__name__}, expected TransformSpec or callable."
            )
        return TransformSpec(
            name=dotted_path,
            needs_planning=bool(getattr(obj, "needs_planning", False)),
            uses_sepa_controller=bool(
                getattr(obj, "uses_sepa_controller", False)
            ),
            compute_context=obj,  # type: ignore[invalid-argument-type]
        )

    raise TypeError(
        f"transform_mode '{dotted_path}' must resolve to TransformSpec "
        f"or callable, got {type(obj).__name__}."
    )


def get_transform_spec(transform_mode: str) -> TransformSpec:
    """Resolve a transform mode to a behavior spec."""
    return _resolve_registry_spec(
        transform_mode,
        builtins=_BUILTIN_TRANSFORM_SPECS,
        cache=_TRANSFORM_SPEC_CACHE,
        load_custom=_load_custom_transform_spec,
        builtin_names=get_builtin_transform_modes,
        config_key="transform_mode",
        custom_label="transforms",
        example="my_module.make_transform_spec",
    )


# ---------------------------------------------------------------------------
