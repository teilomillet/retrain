"""Tree node run command helper."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def run_campaign_node(tree, node_id: str) -> None:
    """Launch a node's campaign and record state."""
    from retrain.campaign.run import run_campaign
    from retrain.tree.model import NodeState
    from retrain.tree.state import save_state

    if node_id not in tree.node_map:
        print(f"Error: unknown node {node_id!r}")
        sys.exit(1)

    node = tree.node_map[node_id]
    ns = tree.state.nodes.setdefault(node_id, NodeState())

    ns.status = "running"
    ns.started_at = datetime.now(timezone.utc).isoformat()
    save_state(tree)

    print(f"Running node {node_id!r}: {node.campaign}")
    campaign_dir = run_campaign(node.campaign)

    ns.campaign_dir = campaign_dir
    ns.status = "done"
    ns.completed_at = datetime.now(timezone.utc).isoformat()
    save_state(tree)

    print(f"Node {node_id!r} campaign completed. Dir: {campaign_dir}")
    print("Run 'retrain tree eval' to evaluate success conditions.")
