"""Tree definition loading and validation."""

from __future__ import annotations

from pathlib import Path

from retrain.tree.model import Tree, TreeNode, parse_success_condition
from retrain.tree.state import load_state

_VALID_METRICS = {"correct_rate", "loss", "mean_reward"}


def load_tree(path: str | Path) -> Tree:
    """Load a tree definition from TOML and its state sidecar."""
    import tomllib

    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    tree_section = data.get("tree", {})
    name = tree_section.get("name", path.stem)
    description = tree_section.get("description", "")

    raw_nodes = tree_section.get("nodes", [])
    nodes: list[TreeNode] = []
    node_map: dict[str, TreeNode] = {}

    for raw in raw_nodes:
        node_id = raw.get("id", "")
        if not node_id:
            raise ValueError(f"Node missing 'id': {raw}")

        campaign = raw.get("campaign", "")
        if not campaign:
            raise ValueError(f"Node {node_id!r} missing 'campaign'")

        sc = None
        success_str = raw.get("success", "")
        if success_str:
            sc = parse_success_condition(success_str)
            if sc.metric not in _VALID_METRICS:
                raise ValueError(
                    f"Node {node_id!r}: unknown metric {sc.metric!r} "
                    f"(valid: {', '.join(sorted(_VALID_METRICS))})"
                )

        node = TreeNode(
            id=node_id,
            label=raw.get("label", node_id),
            campaign=campaign,
            notes=raw.get("notes", ""),
            refs=list(raw.get("refs", [])),
            related=list(raw.get("related", [])),
            success=success_str,
            success_condition=sc,
            on_success=list(raw.get("on_success", [])),
            on_failure=list(raw.get("on_failure", [])),
        )
        nodes.append(node)
        node_map[node_id] = node

    # Validate child references
    for node in nodes:
        for child_id in node.on_success + node.on_failure:
            if child_id not in node_map:
                raise ValueError(
                    f"Node {node.id!r}: references unknown child {child_id!r}"
                )

    state = load_state(path)

    return Tree(
        name=name,
        description=description,
        nodes=nodes,
        node_map=node_map,
        state=state,
        tree_path=path,
    )
