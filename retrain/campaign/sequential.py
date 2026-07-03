"""Sequential in-process campaign execution."""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from retrain.campaign.model import CampaignRun
from retrain.campaign.squeeze import auto_squeeze
from retrain.config import TrainConfig


def run_sequential(
    runs: list[CampaignRun],
    base_config: TrainConfig,
    max_steps: int,
    squeeze_cfg: Mapping[str, object] | None,
) -> tuple[int, int | None]:
    """Execute runs sequentially in-process.

    Returns ``(failed_count, recommended_rank)``.
    """
    from retrain.registry.builtin import get_registry

    failed = 0
    recommended_rank: int | None = None

    for idx, run in enumerate(runs):
        print(f"[{idx + 1}/{len(runs)}] {run['run_name']}")
        try:
            cfg = copy.deepcopy(base_config)
            cfg.advantage_mode = run["advantage_mode"]
            cfg.transform_mode = run["transform_mode"]
            cfg.seed = run["seed"]
            cfg.max_steps = max_steps
            cfg.log_dir = run["log_dir"]
            for key, value in run.get("overrides", {}).items():
                setattr(cfg, key, value)

            # Set wandb fields if configured
            condition = run["condition"]
            if cfg.wandb_project:
                cfg.wandb_group = condition
                cfg.wandb_run_name = run["run_name"]
                cfg.wandb_tags = f"{condition},seed{run['seed']}"

            meta_path = Path(cfg.log_dir) / "run_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(
                    {
                        "trainer": cfg.trainer,
                        "run_id": meta_path.parent.name or "run",
                        "status": "running",
                    }
                )
            )
            runner = get_registry("trainer").create(cfg.trainer, cfg)
            result = runner.run(cfg)
            meta: dict[str, object] = {"trainer": cfg.trainer}
            meta.update(result.to_dict())
            meta_path.write_text(json.dumps(meta))
            if result.ok:
                run["returncode"] = 0
                print("  OK")
            else:
                run["returncode"] = 1
                failed += 1
                print(
                    f"  FAILED: {result.failure_status}"
                    + (f" ({result.error_message})" if result.error_message else "")
                )
                continue

            # Auto-squeeze after first run (errors here don't fail the run)
            if idx == 0 and squeeze_cfg and result.policy_ref:
                try:
                    recommended_rank = auto_squeeze(
                        result.policy_ref,
                        squeeze_cfg,
                        base_config.lora_rank,
                        wandb_project=base_config.wandb_project,
                        wandb_entity=base_config.wandb_entity,
                    )
                except Exception as e:
                    print(f"  Squeeze failed (non-fatal): {e}")
        except RuntimeError as e:
            # Fatal errors (missing backend, bad config) — abort campaign
            print(f"  FATAL: {e}")
            print("\nAborting campaign.")
            sys.exit(1)
        except Exception as e:
            print(f"  FAILED: {e}")
            run["returncode"] = 1
            failed += 1

    return failed, recommended_rank
