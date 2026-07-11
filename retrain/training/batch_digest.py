"""Canonical fingerprints for trainer-logical and local effective rows."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence


_LOGICAL_FORMAT_VERSION = b"retrain.optimizer-batch.v1\0"
_LOCAL_RL_EFFECTIVE_ROWS_FORMAT_VERSION = b"retrain.local-rl-effective-rows.v1\0"
_LOCAL_SFT_EFFECTIVE_ROWS_FORMAT_VERSION = b"retrain.local-sft-effective-rows.v1\0"


def _add_tag(digest: hashlib._Hash, tag: str) -> None:
    encoded = tag.encode("ascii")
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)


def _add_int(digest: hashlib._Hash, value: int) -> None:
    digest.update(struct.pack(">q", value))


def _add_int_rows(
    digest: hashlib._Hash,
    tag: str,
    rows: Sequence[Sequence[int]],
) -> None:
    _add_tag(digest, tag)
    digest.update(struct.pack(">Q", len(rows)))
    for row in rows:
        digest.update(struct.pack(">Q", len(row)))
        for value in row:
            _add_int(digest, int(value))


def _add_float_rows(
    digest: hashlib._Hash,
    tag: str,
    rows: Sequence[Sequence[float]],
    *,
    pack_format: str,
) -> None:
    _add_tag(digest, tag)
    digest.update(struct.pack(">Q", len(rows)))
    for row in rows:
        digest.update(struct.pack(">Q", len(row)))
        for value in row:
            digest.update(struct.pack(pack_format, float(value)))


def _validate_rows(
    tokens: Sequence[Sequence[int]],
    values: Sequence[Sequence[float]],
    *,
    name: str,
) -> None:
    if len(values) != len(tokens):
        raise ValueError(
            f"{name} has {len(values)} rows, expected {len(tokens)}."
        )
    for row_index, (token_row, value_row) in enumerate(zip(tokens, values)):
        if len(value_row) != len(token_row):
            raise ValueError(
                f"{name}[{row_index}] has {len(value_row)} values, "
                f"expected {len(token_row)}."
            )


def _optimizer_batch_sha256(
    datum_tokens: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    token_advantages: Sequence[Sequence[float]],
    *,
    format_version: bytes,
    float_pack_format: str,
    echo_observation_masks: Sequence[Sequence[float]] | None = None,
    echo_full_observation_counts: Sequence[int] | None = None,
    echo_rollout_denominator: int | None = None,
) -> str:

    _validate_rows(datum_tokens, old_logprobs, name="old_logprobs")
    _validate_rows(datum_tokens, token_advantages, name="token_advantages")

    echo_values = (
        echo_observation_masks,
        echo_full_observation_counts,
        echo_rollout_denominator,
    )
    echo_present = echo_observation_masks is not None
    if any(value is not None for value in echo_values) != all(
        value is not None for value in echo_values
    ):
        raise ValueError(
            "ECHO observation masks, full observation counts, and rollout "
            "denominator must be supplied together."
        )
    if echo_present:
        assert echo_observation_masks is not None
        assert echo_full_observation_counts is not None
        assert echo_rollout_denominator is not None
        _validate_rows(
            datum_tokens,
            echo_observation_masks,
            name="echo_observation_masks",
        )
        if len(echo_full_observation_counts) != len(datum_tokens):
            raise ValueError(
                "echo_full_observation_counts has "
                f"{len(echo_full_observation_counts)} rows, expected "
                f"{len(datum_tokens)}."
            )
        if echo_rollout_denominator <= 0:
            raise ValueError("echo_rollout_denominator must be positive.")

    digest = hashlib.sha256(format_version)
    _add_int_rows(digest, "datum_tokens", datum_tokens)
    _add_float_rows(
        digest,
        "old_logprobs",
        old_logprobs,
        pack_format=float_pack_format,
    )
    _add_float_rows(
        digest,
        "token_advantages",
        token_advantages,
        pack_format=float_pack_format,
    )
    _add_tag(digest, "echo_present")
    digest.update(b"\x01" if echo_present else b"\x00")
    if echo_present:
        assert echo_observation_masks is not None
        assert echo_full_observation_counts is not None
        assert echo_rollout_denominator is not None
        _add_float_rows(
            digest,
            "echo_observation_masks",
            echo_observation_masks,
            pack_format=float_pack_format,
        )
        _add_tag(digest, "echo_full_observation_counts")
        digest.update(struct.pack(">Q", len(echo_full_observation_counts)))
        for value in echo_full_observation_counts:
            _add_int(digest, int(value))
        _add_tag(digest, "echo_rollout_denominator")
        _add_int(digest, echo_rollout_denominator)
    return digest.hexdigest()


def logical_optimizer_batch_sha256(
    datum_tokens: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    token_advantages: Sequence[Sequence[float]],
    *,
    echo_observation_masks: Sequence[Sequence[float]] | None = None,
    echo_full_observation_counts: Sequence[int] | None = None,
    echo_rollout_denominator: int | None = None,
) -> str:
    """Hash trainer-logical RL inputs before backend-specific transforms.

    Logical floats use IEEE-754 binary64 framing so even tiny Python-value
    differences remain visible. This digest does not claim that a backend used
    the rows unchanged.
    """

    return _optimizer_batch_sha256(
        datum_tokens,
        old_logprobs,
        token_advantages,
        format_version=_LOGICAL_FORMAT_VERSION,
        float_pack_format=">d",
        echo_observation_masks=echo_observation_masks,
        echo_full_observation_counts=echo_full_observation_counts,
        echo_rollout_denominator=echo_rollout_denominator,
    )


def local_rl_effective_rows_sha256(
    datum_tokens: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    token_advantages: Sequence[Sequence[float]],
    *,
    echo_observation_masks: Sequence[Sequence[float]] | None = None,
    echo_full_observation_counts: Sequence[int] | None = None,
    echo_rollout_denominator: int | None = None,
) -> str:
    """Hash RL rows after local-backend cropping and float32 conversion.

    The local backend materializes logprobs, policy advantages, and ECHO masks
    as float32 tensors. ECHO counts and the rollout denominator remain explicit
    because they scale the joint loss. This deliberately fingerprints rows,
    not loss configuration, model/optimizer state, or an eventual update.
    """

    return _optimizer_batch_sha256(
        datum_tokens,
        old_logprobs,
        token_advantages,
        format_version=_LOCAL_RL_EFFECTIVE_ROWS_FORMAT_VERSION,
        float_pack_format=">f",
        echo_observation_masks=echo_observation_masks,
        echo_full_observation_counts=echo_full_observation_counts,
        echo_rollout_denominator=echo_rollout_denominator,
    )


def local_sft_effective_rows_sha256(
    datum_tokens: Sequence[Sequence[int]],
    target_weights: Sequence[Sequence[float]],
) -> str:
    """Hash cross-entropy SFT rows after local-backend transforms.

    Target weights are framed as the float32 values materialized by the local
    backend. The distinct format version prevents an SFT row from colliding
    with an RL row that happens to contain numerically similar fields.
    """

    _validate_rows(datum_tokens, target_weights, name="target_weights")
    digest = hashlib.sha256(_LOCAL_SFT_EFFECTIVE_ROWS_FORMAT_VERSION)
    _add_int_rows(digest, "datum_tokens", datum_tokens)
    _add_float_rows(
        digest,
        "target_weights",
        target_weights,
        pack_format=">f",
    )
    return digest.hexdigest()


def local_effective_optimizer_batch_sha256(
    datum_tokens: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    token_advantages: Sequence[Sequence[float]],
    *,
    echo_observation_masks: Sequence[Sequence[float]] | None = None,
    echo_full_observation_counts: Sequence[int] | None = None,
    echo_rollout_denominator: int | None = None,
) -> str:
    """Deprecated alias for :func:`local_rl_effective_rows_sha256`."""

    return local_rl_effective_rows_sha256(
        datum_tokens,
        old_logprobs,
        token_advantages,
        echo_observation_masks=echo_observation_masks,
        echo_full_observation_counts=echo_full_observation_counts,
        echo_rollout_denominator=echo_rollout_denominator,
    )


def optimizer_batch_sha256(
    datum_tokens: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    token_advantages: Sequence[Sequence[float]],
    *,
    echo_observation_masks: Sequence[Sequence[float]] | None = None,
    echo_full_observation_counts: Sequence[int] | None = None,
    echo_rollout_denominator: int | None = None,
) -> str:
    """Backward-compatible alias for :func:`logical_optimizer_batch_sha256`."""

    return logical_optimizer_batch_sha256(
        datum_tokens,
        old_logprobs,
        token_advantages,
        echo_observation_masks=echo_observation_masks,
        echo_full_observation_counts=echo_full_observation_counts,
        echo_rollout_denominator=echo_rollout_denominator,
    )
