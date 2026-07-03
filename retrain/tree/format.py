"""ASCII and JSON rendering of the experiment tree."""

from __future__ import annotations

from retrain.tree.model import Tree, node_state
from retrain.tree.status import build_parent_map, effective_status

_STATUS_ICONS: dict[str, str] = {
    "pending": "[ ]",
    "locked": "[#]",
    "running": "[~]",
    "skipped": "[-]",
}

_DONE_STATUS_ICONS: dict[str, str] = {
    "success": "[✓]",
    "failure": "[✗]",
}


def _icon(status: str, outcome: str = "") -> str:
    if status == "done":
        return _DONE_STATUS_ICONS.get(outcome, "[?]")
    return _STATUS_ICONS.get(status, "[?]")


def format_tree(tree: Tree) -> str:
    """Render the tree as an ASCII diagram."""
    parent_map = build_parent_map(tree)
    lines: list[str] = []

    if tree.description:
        lines.append(f"{tree.name}: {tree.description}")
    else:
        lines.append(tree.name)
    lines.append("")

    # Find root nodes (no parents)
    roots = [n for n in tree.nodes if n.id not in parent_map]

    visited: set[str] = set()

    def _render_node(
        node_id: str, prefix: str, child_prefix: str,
    ) -> None:
        if node_id in visited:
            ns = node_state(tree, node_id)
            status = effective_status(node_id, tree, parent_map)
            icon = _icon(status, ns.outcome)
            lines.append(f"{prefix}{icon} {node_id}  (see above)")
            return
        visited.add(node_id)

        node = tree.node_map[node_id]
        ns = node_state(tree, node_id)
        status = effective_status(node_id, tree, parent_map)
        icon = _icon(status, ns.outcome)

        result_str = ""
        if status == "done" and ns.result:
            parts = [f"{k}={v}" for k, v in ns.result.items()]
            result_str = f"  ({', '.join(parts)})"

        lines.append(f"{prefix}{icon} {node.id}  \"{node.label}\"{result_str}")

        for branch, children in (
            ("success", node.on_success),
            ("failure", node.on_failure),
        ):
            for i, cid in enumerate(children):
                p = (
                    f"{child_prefix}{branch} -> "
                    if i == 0
                    else f"{child_prefix}{' ' * len(branch)} -> "
                )
                sub = child_prefix + " " * (len(branch) + 4)
                _render_node(cid, p, sub)

    for root in roots:
        _render_node(root.id, "", "     ")

    return "\n".join(lines)


def format_next(tree: Tree) -> str:
    """List all pending (actionable) nodes with label and notes."""
    parent_map = build_parent_map(tree)
    lines: list[str] = []

    for node in tree.nodes:
        status = effective_status(node.id, tree, parent_map)
        if status == "pending":
            ns = node_state(tree, node.id)
            if ns.status != "running":
                lines.append(f"  {node.id}: {node.campaign}")
                detail = f'"{node.label}"'
                if node.notes:
                    detail += f" — {node.notes}"
                lines.append(f"    {detail}")

    if not lines:
        return "No pending nodes."

    header = "Ready to run:"
    return "\n".join([header] + lines)


def format_show(tree: Tree, node_id: str) -> str:
    """Return a detailed single-node view."""
    if node_id not in tree.node_map:
        raise KeyError(f"Unknown node: {node_id!r}")

    node = tree.node_map[node_id]
    parent_map = build_parent_map(tree)
    ns = node_state(tree, node_id)
    status = effective_status(node_id, tree, parent_map)
    icon = _icon(status, ns.outcome)

    lines: list[str] = [f'{node.id}: "{node.label}"  {icon}']

    lines.append(f"  campaign:    {node.campaign}")
    if node.success:
        lines.append(f"  success:     {node.success}")
    if node.notes:
        lines.append(f"  notes:       {node.notes}")
    if node.refs:
        lines.append(f"  refs:        {', '.join(node.refs)}")
    if node.on_success:
        lines.append(f"  on_success:  {', '.join(node.on_success)}")
    if node.on_failure:
        lines.append(f"  on_failure:  {', '.join(node.on_failure)}")
    if node.related:
        lines.append(f"  related:     {', '.join(node.related)}")
    if ns.campaign_dir:
        lines.append(f"  campaign_dir: {ns.campaign_dir}")
    if ns.result:
        parts = [f"{k}={v}" for k, v in ns.result.items()]
        lines.append(f"  result:      {', '.join(parts)}")
    if ns.annotations:
        lines.append("  annotations:")
        for ann in ns.annotations:
            ts = f"[{ann.at}] " if ann.at else ""
            lines.append(f"    {ts}{ann.text}")

    return "\n".join(lines)


def format_tree_json(tree: Tree) -> dict:
    """Return a JSON-serializable dict of the full tree."""
    parent_map = build_parent_map(tree)
    nodes_out = []
    for node in tree.nodes:
        ns = node_state(tree, node.id)
        status = effective_status(node.id, tree, parent_map)
        nodes_out.append({
            "id": node.id,
            "label": node.label,
            "campaign": node.campaign,
            "status": status,
            "outcome": ns.outcome,
            "success": node.success,
            "on_success": node.on_success,
            "on_failure": node.on_failure,
            "related": node.related,
            "result": ns.result,
            "campaign_dir": ns.campaign_dir,
            "annotations": [
                {"text": a.text, "at": a.at} for a in ns.annotations
            ],
            "notes": node.notes,
            "refs": node.refs,
        })
    return {
        "name": tree.name,
        "description": tree.description,
        "nodes": nodes_out,
    }
