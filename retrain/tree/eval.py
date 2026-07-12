"""Evaluate node success conditions from campaign results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from retrain.tree.model import Tree, node_state
from retrain.tree.state import save_state


def evaluate_node(tree: Tree, node_id: str) -> str | None:
    """Evaluate a node's success condition from campaign results.

    Returns ``"success"``, ``"failure"``, or ``None`` (no condition).
    """
    from retrain.status.scan import scan_campaign

    node = tree.node_map[node_id]
    ns = node_state(tree, node_id)

    if node.success_condition is None:
        return None

    if not ns.campaign_dir:
        raise ValueError(f"Node {node_id!r}: no campaign_dir in state")

    campaign = scan_campaign(Path(ns.campaign_dir))
    if campaign is None:
        raise ValueError(f"Node {node_id!r}: cannot scan campaign at {ns.campaign_dir}")

    metric = node.success_condition.metric
    values: list[float] = []
    for run in campaign.runs:
        if not run.completed:
            continue
        val = getattr(run, metric, None)
        if val is not None:
            values.append(float(val))

    if not values:
        raise ValueError(f"Node {node_id!r}: no completed runs with metric {metric!r}")

    avg = sum(values) / len(values)
    ns.result[metric] = round(avg, 4)

    if node.success_condition.evaluate(avg):
        ns.outcome = "success"
    else:
        ns.outcome = "failure"

    ns.status = "done"
    ns.completed_at = datetime.now(timezone.utc).isoformat()
    save_state(tree)

    return ns.outcome
