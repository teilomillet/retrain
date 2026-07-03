"""Tree note command helper."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def add_note(tree, node_id: str, text: str) -> None:
    """Add an annotation to a node."""
    from retrain.tree.model import Annotation, NodeState
    from retrain.tree.state import save_state

    if node_id not in tree.node_map:
        print(f"Error: unknown node {node_id!r}")
        sys.exit(1)

    ns = tree.state.nodes.setdefault(node_id, NodeState())
    ns.annotations.append(
        Annotation(text=text, at=datetime.now(timezone.utc).isoformat())
    )
    save_state(tree)
    print(f"Note added to {node_id!r}.")
