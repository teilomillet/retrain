from __future__ import annotations

from retrain.environments.timing import collect_observation_timing


class _FakeTrajectoryStep:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def test_collect_observation_timing_from_trajectory_extras() -> None:
    state: dict[str, object] = {
        "trajectory": [
            {
                "extras": {
                    "openenv_info": {
                        "timing": {
                            "dbt_total_s": 1.25,
                            "step_total_s": 1.5,
                            "dbt_target_scoped": True,
                            "action": "Dbt",
                        }
                    }
                }
            }
        ]
    }
    totals: dict[str, float] = {}

    collect_observation_timing(state, totals)

    assert totals == {"dbt_total_s": 1.25, "step_total_s": 1.5}


def test_collect_observation_timing_from_direct_step_timing() -> None:
    state: dict[str, object] = {
        "trajectory": [
            _FakeTrajectoryStep(
                timing={
                    "env_step_s": 0.25,
                    "nan_step_s": float("nan"),
                    "inf_step_s": float("inf"),
                    "flag": True,
                    "action": "Dbt",
                }
            )
        ]
    }
    totals: dict[str, float] = {}

    collect_observation_timing(state, totals)

    assert totals == {"env_step_s": 0.25}
