"""Summary export for test-time discovery."""

from __future__ import annotations

import json
from pathlib import Path

from retrain.training.discovery.archive import DiscoverArchive
from retrain.training.discovery.prompt import truncate_text


def write_discovery_summary(log_dir: Path, archive: DiscoverArchive) -> None:
    best = archive.best_entry()
    top_entries = sorted(
        archive.entries.values(),
        key=lambda e: (e.reward, e.q_value, -e.depth, -e.entry_id),
        reverse=True,
    )[:10]
    payload = {
        "archive_size": len(archive),
        "total_expansions": archive.total_expansions,
        "best_entry_id": best.entry_id,
        "best_reward": best.reward,
        "best_depth": best.depth,
        "best_text": best.text,
        "top_entries": [
            {
                "entry_id": entry.entry_id,
                "parent_id": entry.parent_id,
                "reward": entry.reward,
                "depth": entry.depth,
                "expansions": entry.expansions,
                "text_preview": truncate_text(entry.text, 240),
            }
            for entry in top_entries
        ],
    }
    out_path = log_dir / "ttt_discover.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
