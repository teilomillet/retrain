"""Sampling path for the local backend: engine dispatch + cache policy."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Protocol, cast

from retrain.backends.local import metrics as local_metrics
from retrain.backends.local.checkpointing import configure_gradient_checkpointing
from retrain.backends.local.memory import empty_cuda_cache_if_requested
from retrain.backends.torch import reset_cuda_peak


class _SampleEngine(Protocol):
    def generate(
        self,
        prompt_ids_list: list[list[int]],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        *,
        compute_entropy: bool,
    ) -> list[list[local_metrics.SampleResultLike]]: ...


class _SampleOwner(Protocol):
    engine: object
    cuda_empty_cache: bool

    # Method (not the module function) so subclasses can override the cache
    # policy — UnslothTrainHelper wraps it with its own logits context.
    def _shared_model_sampling_cache_context(self): ...


@contextmanager
def shared_model_cache_context(helper: object):
    """Enable KV cache while sampling on a shared train/infer model.

    Gradient checkpointing forces ``use_cache=False`` on the model config,
    which makes single-model sampling quadratic in generated tokens. Toggle
    checkpointing off (and the cache on) for the duration of sampling, then
    restore the exact checkpointing layer policy afterwards.
    """
    toggled = False
    config = None
    previous_use_cache = None
    if (
        getattr(helper, "gradient_checkpointing", False)
        and getattr(helper, "sample_use_cache", True)
        and not getattr(helper, "_external_engine", False)
        and not getattr(helper, "split_mode", False)
    ):
        model = getattr(helper, "train_model")
        config = getattr(model, "config", None)
        previous_use_cache = getattr(config, "use_cache", None)
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            toggled = True
        if config is not None and previous_use_cache is not None:
            config.use_cache = True
    setattr(helper, "_last_sample_gc_disabled_for_cache", int(toggled))
    try:
        yield
    finally:
        if toggled:
            model = getattr(helper, "train_model")
            setattr(
                helper,
                "_gradient_checkpointing_layer_metrics",
                configure_gradient_checkpointing(
                    model,
                    enabled=True,
                    use_reentrant=getattr(
                        helper,
                        "gradient_checkpointing_use_reentrant",
                        "auto",
                    ),
                    skip_last_n=getattr(
                        helper,
                        "gradient_checkpointing_skip_last_n",
                        0,
                    ),
                ),
            )
            if config is not None and previous_use_cache is not None:
                config.use_cache = previous_use_cache


def sample_groups(
    helper: object,
    prompt_ids_list: list[list[int]],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    *,
    compute_entropy: bool,
) -> list[list[local_metrics.SampleResultLike]]:
    """Run the engine once and return raw SampleResult groups."""
    sample_start = time.perf_counter()
    reset_cuda_peak(
        getattr(helper, "infer_device", getattr(helper, "train_device", "cpu"))
    )
    owner = cast(_SampleOwner, helper)
    engine = cast(_SampleEngine, owner.engine)
    try:
        with owner._shared_model_sampling_cache_context():
            engine_results = engine.generate(
                prompt_ids_list,
                num_samples,
                max_tokens,
                temperature,
                top_p,
                compute_entropy=compute_entropy,
            )
    finally:
        empty_cuda_cache_if_requested(bool(owner.cuda_empty_cache))
    local_metrics.record_sample(
        helper,
        start_s=sample_start,
        prompt_ids_list=prompt_ids_list,
        num_samples=num_samples,
        engine_results=engine_results,
    )
    return engine_results
