"""Planning detector factory dispatched on ``planning_detector``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retrain.planning.regex import RegexPlanningDetector
from retrain.planning.semantic import SemanticPlanningDetector
from retrain.planning.types import PlanningDetector

if TYPE_CHECKING:
    from retrain.config import TrainConfig


_VALID_DETECTORS = {"regex", "semantic"}


def create_planning_detector(config: TrainConfig) -> PlanningDetector:
    """Create the planning detector specified by config."""
    import json

    from retrain.advantages import DEFAULT_STRATEGIC_GRAMS

    detector_type = config.planning_detector

    if detector_type == "regex":
        # Parse strategic grams from config (same logic as trainer.py)
        if config.strategic_grams:
            raw = config.strategic_grams
            if raw.startswith("["):
                grams = [g.strip() for g in json.loads(raw) if g.strip()]
            else:
                grams = [g.strip() for g in raw.split(",") if g.strip()]
        else:
            grams = list(DEFAULT_STRATEGIC_GRAMS)
        return RegexPlanningDetector(grams)

    if detector_type == "semantic":
        return SemanticPlanningDetector(
            model_name=config.planning_model,
            threshold=config.planning_threshold,
        )

    raise ValueError(
        f"Unknown planning_detector '{detector_type}'. "
        f"Choose from: {sorted(_VALID_DETECTORS)}"
    )
