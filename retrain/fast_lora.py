"""Fast LoRA helpers used by the local PyTorch training backend."""

from types import MethodType

import torch


def _config_value(config, key):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def infer_transformer_layer_count(model):
    config = getattr(model, "config", None)
    candidates = [config]
    for nested_key in ("text_config", "language_config", "llm_config"):
        nested = _config_value(config, nested_key)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            value = _config_value(candidate, key)
            if value is None:
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    return 0


class FastLoRALinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        lora_a,
        lora_b,
        scaling,
        detach_lora_input,
        freeze_lora_a,
    ):
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        compute_dtype = x_flat.dtype
        lora_a_compute = lora_a.to(compute_dtype)
        lora_b_compute = lora_b.to(compute_dtype)

        output = torch.matmul(x_flat, weight.to(compute_dtype).T)
        if bias is not None:
            output = output + bias.to(output.dtype)
        lora_hidden = torch.matmul(x_flat, lora_a_compute.T)
        output.addmm_(lora_hidden, lora_b_compute.T, alpha=float(scaling))

        if freeze_lora_a:
            ctx.save_for_backward(weight, lora_a, lora_b, lora_hidden)
            ctx.x_shape = original_shape
        else:
            ctx.save_for_backward(x, weight, lora_a, lora_b)
            ctx.x_shape = None
        ctx.scaling = float(scaling)
        ctx.detach_lora_input = bool(detach_lora_input)
        ctx.freeze_lora_a = bool(freeze_lora_a)
        return output.reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        if ctx.freeze_lora_a:
            weight, lora_a, lora_b, lora_hidden = ctx.saved_tensors
            x = None
            x_shape = ctx.x_shape
        else:
            x, weight, lora_a, lora_b = ctx.saved_tensors
            lora_hidden = None
            x_shape = x.shape
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
        compute_dtype = grad_flat.dtype
        lora_a_compute = lora_a.to(compute_dtype)
        lora_b_compute = lora_b.to(compute_dtype)
        scaling = ctx.scaling

        grad_x = None
        grad_lora_a = None
        grad_lora_b = None
        grad_lora_hidden = grad_flat.matmul(lora_b_compute)

        if ctx.needs_input_grad[0]:
            grad_x = grad_flat.matmul(weight.to(compute_dtype))
            if not ctx.detach_lora_input:
            grad_x = grad_x.addmm(grad_lora_hidden, lora_a_compute, alpha=scaling)
            grad_x = grad_x.reshape(x_shape)
        if ctx.needs_input_grad[3] and not ctx.freeze_lora_a:
            assert x is not None
            x_flat = x.reshape(-1, x.shape[-1])
            grad_lora_a = grad_lora_hidden.T.matmul(x_flat)
            grad_lora_a = (grad_lora_a * scaling).to(lora_a.dtype)
        if ctx.needs_input_grad[4]:
            if lora_hidden is None:
                assert x is not None
                x_flat = x.reshape(-1, x.shape[-1])
                lora_hidden = x_flat.matmul(lora_a_compute.T)
            grad_lora_b = grad_flat.T.matmul(lora_hidden)
            grad_lora_b = (grad_lora_b * scaling).to(lora_b.dtype)

        return grad_x, None, None, grad_lora_a, grad_lora_b, None, None, None


def _mapping_value(container, key, default=None):
    if container is None:
        return default
    try:
        if key in container:
            return container[key]
    except TypeError:
        pass
    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def fast_lora_linear_forward(module, x, *args, **kwargs):
    original_forward = getattr(module, "_retrain_fast_lora_original_forward")
    if args or kwargs:
        return original_forward(x, *args, **kwargs)
    check_args = getattr(module, "_check_forward_args", None)
    if callable(check_args):
        check_args(x, *args, **kwargs)
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return original_forward(x, *args, **kwargs)
    active_adapters = list(getattr(module, "active_adapters", []) or [])
    if len(active_adapters) != 1:
        return original_forward(x, *args, **kwargs)
    adapter = active_adapters[0]
    lora_a_layers = getattr(module, "lora_A", {})
    lora_b_layers = getattr(module, "lora_B", {})
    if adapter not in lora_a_layers or adapter not in lora_b_layers:
        return original_forward(x, *args, **kwargs)
    use_dora = getattr(module, "use_dora", {})
    if bool(_mapping_value(use_dora, adapter, False)):
        return original_forward(x, *args, **kwargs)
    dropout = _mapping_value(getattr(module, "lora_dropout", {}), adapter)
    if dropout is None:
        return original_forward(x, *args, **kwargs)
    dropout_p = float(getattr(dropout, "p", 0.0))
    if dropout_p != 0.0:
        return original_forward(x, *args, **kwargs)
    base_layer = getattr(module, "base_layer", None)
    weight = getattr(base_layer, "weight", None)
    if not torch.is_tensor(weight):
        return original_forward(x, *args, **kwargs)
    bias = getattr(base_layer, "bias", None)
    lora_a = lora_a_layers[adapter].weight
    lora_b = lora_b_layers[adapter].weight
    scaling = _mapping_value(getattr(module, "scaling", {}), adapter)
    if scaling is None:
        return original_forward(x, *args, **kwargs)
    detach_lora_input = bool(getattr(module, "_retrain_fast_lora_detach_input", False))
    freeze_lora_a = bool(getattr(module, "_retrain_fast_lora_freeze_a", False))
    return FastLoRALinearFunction.apply(
        x,
        weight,
        bias,
        lora_a,
        lora_b,
        scaling,
        detach_lora_input,
        freeze_lora_a,
    )


def parse_lora_layers_to_transform(spec, layer_count=0):
    raw = str(spec or "").strip()
    if not raw or raw.lower() in {"all", "default"}:
        return None

    normalized = raw.replace(" ", "").lower()
    if normalized.startswith(("last:", "first:")):
        mode, raw_count = normalized.split(":", 1)
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: expected {mode}:N."
            ) from None
        if count <= 0:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: N must be > 0."
            )
        if layer_count <= 0:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: model layer count is unknown."
            )
        if count > layer_count:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: N exceeds {layer_count} layers."
            )
        if mode == "first":
            return list(range(count))
        return list(range(layer_count - count, layer_count))

    layers: list[int] = []
    for part in normalized.split(","):
        if not part:
            raise ValueError(
                f"Invalid lora_layers_to_transform={raw!r}: empty layer entry."
            )
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad range {part!r}."
                )
            try:
                start, end = (int(bounds[0]), int(bounds[1]))
            except ValueError:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad range {part!r}."
                ) from None
            if start > end:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: range {part!r} is reversed."
                )
            layers.extend(range(start, end + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError:
                raise ValueError(
                    f"Invalid lora_layers_to_transform={raw!r}: bad layer {part!r}."
                ) from None

    if not layers:
        raise ValueError(f"Invalid lora_layers_to_transform={raw!r}: no layers selected.")
    if any(layer < 0 for layer in layers):
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: layer ids must be >= 0."
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: duplicate layer id."
        )
    if layer_count > 0 and max(layers) >= layer_count:
        raise ValueError(
            f"Invalid lora_layers_to_transform={raw!r}: layer id exceeds {layer_count - 1}."
        )
    return layers


def patch_lora_fast_linear_modules(model, *, detach_input: bool, freeze_a: bool) -> int:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return 0

    patched = 0
    for _name, module in named_modules():
        if not hasattr(module, "base_layer"):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if hasattr(module, "_retrain_fast_lora_original_forward"):
            continue
        module._retrain_fast_lora_detach_input = bool(detach_input)
        module._retrain_fast_lora_freeze_a = bool(freeze_a)
        module._retrain_fast_lora_original_forward = module.forward
        module.forward = MethodType(fast_lora_linear_forward, module)
        patched += 1
    return patched
