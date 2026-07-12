"""Fail-closed reset provenance checks for opt-in OpenEnv training guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from retrain.environments.coerce import field as _field


class OpenEnvProvenanceError(RuntimeError):
    """A reset observation did not match the configured training task set."""


@dataclass(frozen=True)
class ResetProvenance:
    """Task identity declared by one OpenEnv reset observation."""

    task_id: str | None
    task_source: str | None


@dataclass(frozen=True)
class ResetProvenanceGuard:
    """Optional exact source and task-ID expectations for OpenEnv resets."""

    expected_task_source: str | None = None
    expected_task_ids: frozenset[str] | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.expected_task_source is not None or self.expected_task_ids is not None
        )

    def validate(
        self,
        observation: object,
        *,
        context: str,
        expected_identity: ResetProvenance | None = None,
    ) -> ResetProvenance:
        """Validate one reset and return its unambiguous declared identity."""
        provenance = extract_reset_provenance(observation, context=context)
        if self.expected_task_source is not None:
            if provenance.task_source is None:
                raise OpenEnvProvenanceError(
                    f"{context} is missing task_source; expected "
                    f"{self.expected_task_source!r}."
                )
            if provenance.task_source != self.expected_task_source:
                raise OpenEnvProvenanceError(
                    f"{context} task_source mismatch: expected "
                    f"{self.expected_task_source!r}, got {provenance.task_source!r}."
                )
        if self.expected_task_ids is not None:
            if provenance.task_id is None:
                raise OpenEnvProvenanceError(
                    f"{context} is missing task_id; an expected_task_ids guard is active."
                )
            if provenance.task_id not in self.expected_task_ids:
                raise OpenEnvProvenanceError(
                    f"{context} returned unexpected task_id {provenance.task_id!r}."
                )

        if expected_identity is not None:
            _validate_identity(provenance, expected_identity, context=context)
        return provenance

    def validate_exact_task_set(
        self,
        provenances: Sequence[ResetProvenance],
        *,
        context: str,
    ) -> None:
        """Require preload seeds to cover exactly the configured task-ID set."""
        if self.expected_task_ids is None:
            return
        observed = {item.task_id for item in provenances if item.task_id is not None}
        if observed == self.expected_task_ids:
            return
        missing = sorted(self.expected_task_ids - observed)
        unexpected = sorted(observed - self.expected_task_ids)
        raise OpenEnvProvenanceError(
            f"{context} task-ID set mismatch: missing={missing}, unexpected={unexpected}."
        )


def extract_reset_provenance(
    observation: object,
    *,
    context: str,
) -> ResetProvenance:
    """Read consistent task identity from top-level, info, or metadata fields."""
    containers = _provenance_containers(observation)
    task_ids = _collect_values(containers, "task_id", context=context, task_id=True)
    task_sources = _collect_values(
        containers,
        "task_source",
        context=context,
        task_id=False,
    )
    return ResetProvenance(
        task_id=_unique_value(task_ids, "task_id", context=context),
        task_source=_unique_value(task_sources, "task_source", context=context),
    )


def parse_expected_task_source(raw: object) -> str | None:
    """Parse an optional non-empty exact task source from environment args."""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("expected_task_source must be a non-empty string.")
    return raw.strip()


def parse_expected_task_ids(raw: object) -> frozenset[str] | None:
    """Parse an optional duplicate-free list of exact task IDs."""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected_task_ids must be a non-empty list of strings.")
    values: list[str] = []
    for value in cast(list[object], raw):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected_task_ids must contain only non-empty strings.")
        values.append(value.strip())
    if len(set(values)) != len(values):
        raise ValueError("expected_task_ids must not contain duplicates.")
    return frozenset(values)


def _provenance_containers(observation: object) -> list[tuple[str, object]]:
    containers = [("observation", observation)]
    for name in ("info", "metadata"):
        nested = _field(observation, name)
        if nested is not None:
            containers.append((name, nested))
    return containers


def _collect_values(
    containers: Sequence[tuple[str, object]],
    field_name: str,
    *,
    context: str,
    task_id: bool,
) -> list[str]:
    values: list[str] = []
    for container_name, container in containers:
        raw = _field(container, field_name)
        if raw is None or raw == "":
            continue
        if task_id and isinstance(raw, int) and not isinstance(raw, bool):
            values.append(str(raw))
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise OpenEnvProvenanceError(
                f"{context} has invalid {container_name}.{field_name}; "
                "expected a non-empty string."
            )
        values.append(raw.strip())
    return values


def _unique_value(
    values: Sequence[str], field_name: str, *, context: str
) -> str | None:
    unique = set(values)
    if len(unique) > 1:
        raise OpenEnvProvenanceError(
            f"{context} has conflicting {field_name} values: {sorted(unique)}."
        )
    return next(iter(unique), None)


def _validate_identity(
    observed: ResetProvenance,
    expected: ResetProvenance,
    *,
    context: str,
) -> None:
    if expected.task_id is not None and observed.task_id != expected.task_id:
        raise OpenEnvProvenanceError(
            f"{context} changed task_id for the preload seed: expected "
            f"{expected.task_id!r}, got {observed.task_id!r}."
        )
    if (
        expected.task_source is not None
        and observed.task_source != expected.task_source
    ):
        raise OpenEnvProvenanceError(
            f"{context} changed task_source for the preload seed: expected "
            f"{expected.task_source!r}, got {observed.task_source!r}."
        )
