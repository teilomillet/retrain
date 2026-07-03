"""Derive node status from parent outcomes."""

from __future__ import annotations

from retrain.tree.model import Tree, node_state


def build_parent_map(
    tree: Tree,
) -> dict[str, list[tuple[str, str]]]:
    """Build ``{child_id: [(parent_id, "success"|"failure"), ...]}``."""
    parents: dict[str, list[tuple[str, str]]] = {}
    for node in tree.nodes:
        for child_id in node.on_success:
            parents.setdefault(child_id, []).append((node.id, "success"))
        for child_id in node.on_failure:
            parents.setdefault(child_id, []).append((node.id, "failure"))
    return parents


def effective_status(
    node_id: str,
    tree: Tree,
    parent_map: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    """Derive the effective status for a node."""
    ns = node_state(tree, node_id)
    # Explicit statuses override derivation
    if ns.status in ("running", "done", "skipped"):
        return ns.status

    if parent_map is None:
        parent_map = build_parent_map(tree)

    parents = parent_map.get(node_id, [])
    if not parents:
        # Root node
        return "pending"

    # Check each parent edge
    for parent_id, branch in parents:
        pstate = node_state(tree, parent_id)
        if pstate.status == "done" and pstate.outcome == branch:
            return "pending"

    # Check if any parent is done but on wrong branch → skipped
    all_parents_decided = True
    for parent_id, branch in parents:
        pstate = node_state(tree, parent_id)
        if pstate.status == "done":
            if pstate.outcome != branch:
                continue  # wrong branch
        else:
            all_parents_decided = False

    if all_parents_decided:
        return "skipped"

    return "locked"
