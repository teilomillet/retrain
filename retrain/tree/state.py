"""Tree state persistence (JSON sidecar beside the tree TOML)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from retrain.tree.model import Annotation, NodeState, Tree, TreeState, node_state


def _state_path(tree_path: Path) -> Path:
    """State JSON lives beside the tree TOML."""
    return tree_path.parent / "tree_state.json"


def load_state(tree_path: Path) -> TreeState:
    """Load state from JSON, graceful fallback on missing/corrupt."""
    sp = _state_path(tree_path)
    if not sp.exists():
        return TreeState()
    try:
        raw = json.loads(sp.read_text())
    except (json.JSONDecodeError, OSError):
        return TreeState()

    nodes: dict[str, NodeState] = {}
    for nid, nraw in raw.get("nodes", {}).items():
        annotations = [
            Annotation(text=a.get("text", ""), at=a.get("at", ""))
            for a in nraw.get("annotations", [])
        ]
        nodes[nid] = NodeState(
            status=nraw.get("status", "pending"),
            outcome=nraw.get("outcome", ""),
            campaign_dir=nraw.get("campaign_dir", ""),
            result=nraw.get("result", {}),
            started_at=nraw.get("started_at", ""),
            completed_at=nraw.get("completed_at", ""),
            annotations=annotations,
        )
    return TreeState(nodes=nodes)


def save_state(tree: Tree) -> None:
    """Atomic write of tree state to JSON (tmp + rename)."""
    sp = _state_path(tree.tree_path)
    sp.parent.mkdir(parents=True, exist_ok=True)

    out: dict = {"nodes": {}}
    for nid, ns in tree.state.nodes.items():
        out["nodes"][nid] = {
            "status": ns.status,
            "outcome": ns.outcome,
            "campaign_dir": ns.campaign_dir,
            "result": ns.result,
            "started_at": ns.started_at,
            "completed_at": ns.completed_at,
            "annotations": [
                {"text": a.text, "at": a.at} for a in ns.annotations
            ],
        }

    fd, tmp = tempfile.mkstemp(dir=str(sp.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        os.replace(tmp, str(sp))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def reset_node(tree: Tree, node_id: str) -> None:
    """Reset a node to re-runnable state. Preserves annotations."""
    if node_id not in tree.node_map:
        raise KeyError(f"Unknown node: {node_id!r}")

    ns = node_state(tree, node_id)
    ns.status = "pending"
    ns.outcome = ""
    ns.campaign_dir = ""
    ns.result = {}
    ns.started_at = ""
    ns.completed_at = ""
    # Preserve annotations — they're a lab notebook
    save_state(tree)
