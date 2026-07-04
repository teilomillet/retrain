#!/usr/bin/env python3
"""Run a cheap live W&B checkpoint-artifact recovery drill.

The drill avoids GPU/model downloads by patching in a tiny fake SFT backend and
tokenizer, but it uses retrain's real SFT runner, real checkpoint artifact code,
and the real W&B artifact service. It proves the preemption story:

1. train two tiny SFT steps with checkpoint_artifacts="wandb";
2. download the first periodic checkpoint artifact alias;
3. delete the original local log/adapter paths;
4. resume from the downloaded artifact-local adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROJECT = "retrain-recovery-drill"


def _bool_arg(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cheap live W&B artifact round-trip and resume drill for "
            "retrain SFT checkpoints."
        )
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("RETRAIN_WANDB_RECOVERY_PROJECT")
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_PROJECT,
        help="W&B project for the drill.",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("RETRAIN_WANDB_RECOVERY_ENTITY")
        or os.environ.get("WANDB_ENTITY")
        or "",
        help="Optional W&B entity/team.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional W&B run name. Defaults to drill-<random>.",
    )
    parser.add_argument(
        "--root",
        default="",
        help="Working directory. Defaults to a new /tmp/retrain-wandb-recovery-* dir.",
    )
    parser.add_argument(
        "--cleanup",
        type=_bool_arg,
        default=False,
        help="Delete the drill root on success.",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume-max-steps", type=int, default=2)
    parser.add_argument("--poll-attempts", type=int, default=12)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON evidence output path.",
    )
    return parser


@dataclass
class FakeSftHelper:
    sft_loss_fn: str = ""
    calls: list[dict[str, object]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    saved: list[tuple[str, str]] = field(default_factory=list)
    shutdown_called: bool = False

    def sft_train_step(
        self,
        all_tokens: list[list[int]],
        all_advantages: list[list[float]],
        lr: float,
        weight_decay: float,
    ) -> float:
        self.calls.append(
            {
                "tokens": all_tokens,
                "advantages": all_advantages,
                "lr": lr,
                "weight_decay": weight_decay,
                "loss_fn": self.sft_loss_fn,
            }
        )
        return 0.25

    def save_adapter(self, path: str, name: str) -> str:
        save_dir = Path(path) / name
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "adapter_model.safetensors").write_text(
            f"fake adapter {name}\n"
        )
        (save_dir / "adapter_config.json").write_text(
            json.dumps({"name": name}) + "\n"
        )
        self.saved.append((path, name))
        return str(save_dir)

    def load_state(self, ref: str) -> None:
        self.loaded.append(ref)

    def runtime_metrics(self) -> dict[str, float]:
        return {"fake_metric": 1.0}

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeBackendRegistry:
    def __init__(self, helper: FakeSftHelper) -> None:
        self.helper = helper

    def create(self, name: str, config: object) -> FakeSftHelper:
        _ = config
        if name != "unsloth":
            raise ValueError(f"unexpected fake backend name: {name}")
        return self.helper


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        _ = tokenize
        rendered = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
        if add_generation_prompt:
            rendered += "assistant:"
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]


class WandbRunLike(Protocol):
    entity: str
    project: str
    id: str
    name: str


def _patch_fake_training_stack(helper: FakeSftHelper) -> None:
    import transformers

    setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    import retrain.registry.builtin as builtin

    def fake_get_registry(name: str) -> FakeBackendRegistry:
        if name != "backend":
            raise ValueError(f"unexpected registry: {name}")
        return FakeBackendRegistry(helper)

    setattr(builtin, "get_registry", fake_get_registry)


def _capture_wandb_run() -> dict[str, WandbRunLike]:
    import wandb

    real_init = cast(Callable[..., object], wandb.init)
    captured: dict[str, WandbRunLike] = {}

    def init_capture(*args: object, **kwargs: object) -> WandbRunLike:
        run = cast(WandbRunLike, real_init(*args, **kwargs))
        captured["run"] = run
        return run

    setattr(wandb, "init", init_capture)
    return captured


def _artifact_name(run_id: str) -> str:
    return f"retrain-{run_id}-checkpoints"


def _artifact_ref(
    *,
    entity: str,
    project: str,
    run_id: str,
    alias: str,
) -> str:
    return f"{entity}/{project}/{_artifact_name(run_id)}:{alias}"


def _download_artifact(
    artifact_ref: str,
    *,
    root: Path,
    poll_attempts: int,
    poll_seconds: float,
) -> Path:
    import wandb

    api = wandb.Api()
    last_error: Exception | None = None
    for _attempt in range(1, poll_attempts + 1):
        try:
            artifact = api.artifact(artifact_ref, type="retrain_checkpoint")
            return Path(artifact.download(root=str(root)))
        except Exception as exc:
            last_error = exc
            time.sleep(poll_seconds)
    raise RuntimeError(f"W&B artifact did not become available: {last_error}")


def _assert_downloaded_artifact(downloaded: Path) -> None:
    required = [
        downloaded / "trainer_state.json",
        downloaded / "latest_sampler_path.txt",
        downloaded / "adapter",
        downloaded / "adapter" / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"downloaded artifact is missing required files: {missing}")


def _load_wandb_module() -> ModuleType:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install with: "
            "uv pip install --python .venv/bin/python 'retrain[wandb]'"
        ) from exc
    return wandb


def run_drill(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.resume_max_steps <= 1:
        raise ValueError("--resume-max-steps must be > 1")

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(
        prefix="retrain-wandb-recovery-"
    ))
    root.mkdir(parents=True, exist_ok=True)
    if args.cleanup and args.output:
        output_path = Path(args.output)
        root_resolved = root.resolve()
        output_resolved = output_path.resolve()
        if output_resolved == root_resolved or output_resolved.is_relative_to(
            root_resolved
        ):
            raise ValueError(
                "--output must be outside --root when --cleanup true because "
                "cleanup deletes the drill root."
            )

    _load_wandb_module()
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")

    run_name = args.run_name or f"drill-{uuid.uuid4().hex[:8]}"

    helper = FakeSftHelper()
    _patch_fake_training_stack(helper)
    captured = _capture_wandb_run()

    from retrain.config import TrainConfig
    from retrain.training.runner import SftRunner
    from retrain.training.state import load_trainer_state

    data_path = root / "sft.jsonl"
    data_path.write_text(json.dumps({"text": "hello recovery"}) + "\n")
    adapter_path = root / "adapter"
    log_dir = root / "logs"

    config = TrainConfig(
        trainer="sft",
        backend="unsloth",
        sft_data_path=str(data_path),
        sft_batch_size=1,
        max_steps=args.steps,
        save_every=1,
        batch_size=1,
        lr=1e-4,
        model="fake-model",
        adapter_path=str(adapter_path),
        log_dir=str(log_dir),
        wandb_project=args.project,
        wandb_entity=args.entity,
        wandb_run_name=run_name,
        checkpoint_artifacts="wandb",
    )

    result = SftRunner().run(config)
    if not result.ok:
        raise RuntimeError(
            f"initial SFT run failed: {result.failure_status} {result.error_message}"
        )

    run = captured.get("run")
    if run is None:
        raise RuntimeError("W&B run was not captured")

    checkpoint_alias = "checkpoint_step_1"
    artifact_ref = _artifact_ref(
        entity=run.entity,
        project=run.project,
        run_id=run.id,
        alias=checkpoint_alias,
    )
    download_dir = root / "downloaded_artifact"
    downloaded = _download_artifact(
        artifact_ref,
        root=download_dir,
        poll_attempts=args.poll_attempts,
        poll_seconds=args.poll_seconds,
    )
    _assert_downloaded_artifact(downloaded)

    shutil.rmtree(log_dir)
    shutil.rmtree(adapter_path)
    state = load_trainer_state(str(downloaded))
    expected_adapter = str(downloaded / "adapter")
    if state.get("checkpoint_path") != expected_adapter:
        raise RuntimeError(
            "trainer_state did not fall back to the artifact-local adapter: "
            f"{state.get('checkpoint_path')} != {expected_adapter}"
        )

    resume_helper = FakeSftHelper()
    _patch_fake_training_stack(resume_helper)
    resume_config = TrainConfig(
        trainer="sft",
        backend="unsloth",
        sft_data_path=str(data_path),
        sft_batch_size=1,
        max_steps=args.resume_max_steps,
        save_every=1,
        batch_size=1,
        lr=1e-4,
        model="fake-model",
        adapter_path=str(root / "resume_adapter"),
        log_dir=str(root / "resume_logs"),
        resume_from=str(downloaded),
        checkpoint_artifacts="off",
    )
    resume_result = SftRunner().run(resume_config)
    if not resume_result.ok:
        raise RuntimeError(
            f"resume failed: {resume_result.failure_status} "
            f"{resume_result.error_message}"
        )
    if resume_helper.loaded != [expected_adapter]:
        raise RuntimeError(
            f"resume loaded {resume_helper.loaded}, expected {[expected_adapter]}"
        )
    if len(resume_helper.calls) != args.resume_max_steps - 1:
        raise RuntimeError(
            f"resume ran {len(resume_helper.calls)} step(s), expected "
            f"{args.resume_max_steps - 1}"
        )

    evidence: dict[str, object] = {
        "status": "succeeded",
        "root": str(root),
        "wandb_entity": run.entity,
        "wandb_project": run.project,
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
        "artifact_ref": artifact_ref,
        "downloaded_artifact": str(downloaded),
        "checkpoint_alias": checkpoint_alias,
        "checkpoint_state_step": state["step"],
        "checkpoint_state_name": state["checkpoint_name"],
        "artifact_local_adapter": expected_adapter,
        "original_log_dir_deleted": not log_dir.exists(),
        "original_adapter_path_deleted": not adapter_path.exists(),
        "initial_saved": helper.saved,
        "resume_loaded": resume_helper.loaded,
        "resume_steps": len(resume_helper.calls),
        "resume_policy_ref": resume_result.policy_ref,
    }
    if args.cleanup:
        shutil.rmtree(root)
        evidence["root_cleaned"] = True
    else:
        evidence["root_cleaned"] = False
    return evidence


def main() -> int:
    args = _build_parser().parse_args()
    try:
        evidence = run_drill(args)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        if args.output:
            Path(args.output).write_text(json.dumps(evidence, indent=2) + "\n")
        print("DRILL_OK real_wandb_checkpoint_artifact_resume_succeeded")
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
