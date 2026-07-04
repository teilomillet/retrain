"""Checkpoint recoverability helpers for local and W&B-backed runs."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from retrain.config import TrainConfig
from retrain.training.state import TRAINER_STATE_FILE


_OFFLINE_WANDB_MODES = {"disabled", "dryrun", "offline"}
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_LOG_ARTIFACT_FILES = (
    TRAINER_STATE_FILE,
    "latest_sampler_path.txt",
    "resolved_config.json",
    "sft_data.snapshot.jsonl",
    "sft_data_recoverability.json",
    "sft_manifest.json",
)


class WandbArtifactLike(Protocol):
    def add_file(self, local_path: str, *, name: str | None = None) -> object: ...

    def add_dir(self, local_path: str, *, name: str | None = None) -> object: ...


class WandbRunArtifactLike(Protocol):
    def log(
        self,
        data: Mapping[str, object],
        *,
        step: int | None = None,
    ) -> object: ...

    def log_artifact(
        self,
        artifact: WandbArtifactLike,
        *,
        aliases: list[str] | None = None,
    ) -> object: ...


class WandbArtifactModuleLike(Protocol):
    def Artifact(
        self,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, object] | None = None,
    ) -> WandbArtifactLike: ...


@dataclass(frozen=True)
class WandbLiveStatus:
    live: bool
    reason: str


@dataclass(frozen=True)
class CheckpointArtifactResult:
    uploaded: bool
    artifact_name: str = ""
    reason: str = ""


def checkpoint_artifacts_enabled(
    config: TrainConfig,
    wandb_run: object | None,
) -> bool:
    """Return whether checkpoint artifact upload should be attempted."""
    return config.checkpoint_artifacts != "off" and wandb_run is not None


def wandb_live_status(wandb_run: object | None) -> WandbLiveStatus:
    """Detect whether W&B writes are live enough to survive ephemeral disk loss."""
    if wandb_run is None:
        return WandbLiveStatus(live=False, reason="wandb_disabled")

    env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if env_mode in _OFFLINE_WANDB_MODES:
        return WandbLiveStatus(live=False, reason=f"WANDB_MODE={env_mode}")

    run_mode = _string_attr(wandb_run, "mode")
    settings = getattr(wandb_run, "settings", None)
    settings_mode = _string_attr(settings, "mode") if settings is not None else ""
    for mode in (run_mode, settings_mode):
        if mode.lower() in _OFFLINE_WANDB_MODES:
            return WandbLiveStatus(live=False, reason=f"wandb mode={mode}")

    return WandbLiveStatus(live=True, reason="online")


def checkpoint_recoverability_wandb_metrics(
    config: TrainConfig,
    wandb_run: object | None,
) -> dict[str, int | str]:
    """Small heartbeat that makes checkpoint durability visible in W&B."""
    enabled = checkpoint_artifacts_enabled(config, wandb_run)
    live = wandb_live_status(wandb_run)
    durable = enabled and live.live
    periodic = config.save_every > 0
    return {
        "train/recoverability/checkpoint_artifacts_enabled": int(enabled),
        "train/recoverability/checkpoint_artifacts_required": int(
            config.checkpoint_artifacts == "wandb"
        ),
        "train/recoverability/checkpoint_artifacts_live": int(durable),
        "train/recoverability/periodic_checkpoints_enabled": int(periodic),
        "train/recoverability/preemption_resume_ready": int(durable and periodic),
        "train/recoverability/local_only": int(not durable),
        "train/recoverability/save_every": int(config.save_every),
        "train/recoverability/checkpoint_artifacts_mode": config.checkpoint_artifacts,
        "train/recoverability/checkpoint_artifacts_status": live.reason,
    }


def announce_checkpoint_recoverability(
    config: TrainConfig,
    wandb_run: object | None,
) -> None:
    """Print one explicit durability status line at run start."""
    if config.checkpoint_artifacts == "off":
        print("Checkpoint artifacts: off (checkpoints stay local only).")
        return
    if wandb_run is None:
        print(
            "WARNING: checkpoint artifacts are local-only. Set "
            "[logging] wandb_project to mirror saved checkpoints to W&B "
            "Artifacts, or checkpoint_artifacts = 'off' to silence this."
        )
        return

    status = wandb_live_status(wandb_run)
    if not status.live:
        message = (
            "W&B checkpoint artifacts are not live "
            f"({status.reason}); ephemeral disk loss can still lose checkpoints."
        )
        if config.checkpoint_artifacts == "wandb":
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return

    if config.save_every <= 0:
        message = (
            "checkpoint artifacts are live, but save_every=0 means only the "
            "final adapter will be uploaded. Spot/preemptible runs cannot "
            "resume mid-run; set save_every > 0."
        )
        if config.checkpoint_artifacts == "wandb":
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        print("Checkpoint artifacts: W&B Artifacts (final adapter only).")
        return

    print("Checkpoint artifacts: W&B Artifacts (saved checkpoints + final adapter).")


def upload_checkpoint_artifact(
    config: TrainConfig,
    wandb_run: object | None,
    *,
    checkpoint_name: str,
    checkpoint_path: str | None,
    step: int,
) -> CheckpointArtifactResult:
    """Upload a saved checkpoint/final adapter plus recoverability files to W&B."""
    if config.checkpoint_artifacts == "off":
        return CheckpointArtifactResult(uploaded=False, reason="disabled")
    if wandb_run is None:
        if config.checkpoint_artifacts == "wandb":
            raise RuntimeError(
                "checkpoint_artifacts='wandb' requires an active W&B run."
            )
        return CheckpointArtifactResult(uploaded=False, reason="wandb_disabled")

    status = wandb_live_status(wandb_run)
    if not status.live:
        message = (
            f"checkpoint artifact {checkpoint_name!r} was not uploaded because "
            f"W&B is not live ({status.reason})."
        )
        if config.checkpoint_artifacts == "wandb":
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        _log_checkpoint_upload_status(
            wandb_run,
            step=step,
            checkpoint_name=checkpoint_name,
            uploaded=False,
            reason=status.reason,
            artifact_name="",
        )
        return CheckpointArtifactResult(uploaded=False, reason=status.reason)

    try:
        import wandb as wandb_module

        wandb = cast(WandbArtifactModuleLike, wandb_module)
        artifact_name = _checkpoint_artifact_name(config, wandb_run)
        payload_kind, resolved_checkpoint_path = _checkpoint_payload_metadata(
            checkpoint_path
        )
        artifact = wandb.Artifact(
            artifact_name,
            "retrain_checkpoint",
            metadata={
                "kind": "retrain_checkpoint",
                "checkpoint_name": checkpoint_name,
                "step": int(step),
                "checkpoint_path": checkpoint_path or "",
                "checkpoint_payload": payload_kind,
                "resolved_checkpoint_path": resolved_checkpoint_path,
                "log_dir": str(Path(config.log_dir).expanduser().resolve()),
                "adapter_path": config.adapter_path,
                "trainer": config.trainer,
                "backend": config.backend,
                "model": config.model,
                "save_every": int(config.save_every),
                "resume_hint": f"retrain --resume {Path(config.log_dir).expanduser().resolve()}",
            },
        )
        _add_checkpoint_payload(artifact, checkpoint_path)
        _add_recoverability_files(artifact, Path(config.log_dir))
        aliases = _checkpoint_aliases(checkpoint_name)
        run_with_artifacts = cast(WandbRunArtifactLike, wandb_run)
        run_with_artifacts.log_artifact(artifact, aliases=aliases)
        _log_checkpoint_upload_status(
            wandb_run,
            step=step,
            checkpoint_name=checkpoint_name,
            uploaded=True,
            reason="uploaded",
            artifact_name=artifact_name,
        )
        print(f"W&B checkpoint artifact uploaded: {artifact_name} ({checkpoint_name})")
        return CheckpointArtifactResult(uploaded=True, artifact_name=artifact_name)
    except Exception as exc:
        message = f"W&B checkpoint artifact upload failed for {checkpoint_name}: {exc}"
        if config.checkpoint_artifacts == "wandb":
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}")
        _log_checkpoint_upload_status(
            wandb_run,
            step=step,
            checkpoint_name=checkpoint_name,
            uploaded=False,
            reason=type(exc).__name__,
            artifact_name="",
        )
        return CheckpointArtifactResult(
            uploaded=False,
            reason=type(exc).__name__,
        )


def _checkpoint_payload_metadata(checkpoint_path: str | None) -> tuple[str, str]:
    if not checkpoint_path:
        return "none", ""
    path = Path(checkpoint_path).expanduser()
    if path.exists():
        return "local_dir" if path.is_dir() else "local_file", str(path.resolve())
    if _URI_RE.match(checkpoint_path):
        return "remote_ref", checkpoint_path
    return "missing_local_path", str(path)


def _add_checkpoint_payload(
    artifact: WandbArtifactLike,
    checkpoint_path: str | None,
) -> None:
    if not checkpoint_path or _URI_RE.match(checkpoint_path):
        return
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
    if path.is_dir():
        artifact.add_dir(str(path), name="adapter")
    else:
        artifact.add_file(str(path), name=f"adapter/{path.name}")


def _add_recoverability_files(artifact: WandbArtifactLike, log_dir: Path) -> None:
    for file_name in _LOG_ARTIFACT_FILES:
        path = log_dir / file_name
        if path.is_file():
            artifact.add_file(str(path), name=file_name)


def _log_checkpoint_upload_status(
    wandb_run: object,
    *,
    step: int,
    checkpoint_name: str,
    uploaded: bool,
    reason: str,
    artifact_name: str,
) -> None:
    run = cast(WandbRunArtifactLike, wandb_run)
    try:
        run.log(
            {
                "train/recoverability/latest_checkpoint_uploaded": int(uploaded),
                "train/recoverability/latest_checkpoint_step": int(step),
                "train/recoverability/latest_checkpoint_name": checkpoint_name,
                "train/recoverability/latest_checkpoint_reason": reason,
                "train/recoverability/latest_checkpoint_artifact": artifact_name,
            },
            step=step,
        )
    except Exception:
        pass


def _checkpoint_artifact_name(config: TrainConfig, wandb_run: object) -> str:
    run_slug = (
        _string_attr(wandb_run, "id")
        or _string_attr(wandb_run, "name")
        or config.wandb_run_name
        or Path(config.log_dir).name
        or "run"
    )
    return _sanitize_name(f"retrain-{run_slug}-checkpoints", max_len=128)


def _checkpoint_aliases(checkpoint_name: str) -> list[str]:
    aliases = ["latest", _sanitize_name(checkpoint_name, max_len=64)]
    if checkpoint_name == "final":
        aliases.append("final")
    return list(dict.fromkeys(aliases))


def _sanitize_name(value: str, *, max_len: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    return cleaned or "checkpoint"


def _string_attr(obj: object, attr: str) -> str:
    value = getattr(obj, attr, "")
    return value if isinstance(value, str) else ""
