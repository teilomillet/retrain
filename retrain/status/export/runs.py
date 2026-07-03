"""JSON rendering for exported run snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from retrain.status.export.types import RunSnapshot


def _render_runs_json(root: Path, snapshots: list[RunSnapshot], *, generated_at: float) -> str:
    return json.dumps(
        {
            "generated_at": generated_at,
            "root": str(root),
            "runs": [snapshot.to_dict() for snapshot in snapshots],
        },
        indent=2,
        sort_keys=True,
    )
