"""Experiment tech tree: dependency graph for experiment campaigns."""

from __future__ import annotations

import json
import operator
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_OP_MAP = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}

_CONDITION_RE = re.compile(r"^(\w+)\s*(>=|<=|>|<|==)\s*([0-9.]+)$")


@dataclass
class SuccessCondition:
    """Parsed success condition like ``correct_rate > 0.55``."""

    metric: str
    op: str
    threshold: float

    def evaluate(self, value: float) -> bool:
        return _OP_MAP[self.op](value, self.threshold)


def parse_success_condition(text: str) -> SuccessCondition:
    """Parse ``"metric op value"`` into a :class:`SuccessCondition`."""
    m = _CONDITION_RE.match(text.strip())
    if not m:
        raise ValueError(f"Invalid success condition: {text!r}")
    return SuccessCondition(
        metric=m.group(1),
        op=m.group(2),
        threshold=float(m.group(3)),
    )


@dataclass
class TreeNode:
    """A single experiment node in the tech tree."""

    id: str
    label: str
    campaign: str = ""
    notes: str = ""
    refs: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    success: str = ""
    success_condition: SuccessCondition | None = None
    on_success: list[str] = field(default_factory=list)
    on_failure: list[str] = field(default_factory=list)


@dataclass
class Annotation:
    """Timestamped annotation on a node."""

    text: str
    at: str = ""


@dataclass
class NodeState:
    """Runtime state for a single node."""

    status: str = "pending"  # pending | locked | running | done | skipped
    outcome: str = ""  # success | failure (only when status == done)
    campaign_dir: str = ""
    result: dict[str, float] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class TreeState:
    """Persisted state for all nodes."""

    nodes: dict[str, NodeState] = field(default_factory=dict)


@dataclass
class Tree:
    """Full experiment tree: definition + state."""

    name: str
    description: str
    nodes: list[TreeNode]
    node_map: dict[str, TreeNode]
    state: TreeState
    tree_path: Path


# ---------------------------------------------------------------------------
# TOML loading + validation
# ---------------------------------------------------------------------------

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

    state = _load_state(path)

    return Tree(
        name=name,
        description=description,
        nodes=nodes,
        node_map=node_map,
        state=state,
        tree_path=path,
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def _state_path(tree_path: Path) -> Path:
    """State JSON lives beside the tree TOML."""
    return tree_path.parent / "tree_state.json"


def _load_state(tree_path: Path) -> TreeState:
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


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


def _build_parent_map(
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


def _get_node_state(tree: Tree, node_id: str) -> NodeState:
    """Get or create node state."""
    if node_id not in tree.state.nodes:
        tree.state.nodes[node_id] = NodeState()
    return tree.state.nodes[node_id]


def effective_status(
    node_id: str,
    tree: Tree,
    parent_map: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    """Derive the effective status for a node."""
    ns = _get_node_state(tree, node_id)
    # Explicit statuses override derivation
    if ns.status in ("running", "done", "skipped"):
        return ns.status

    if parent_map is None:
        parent_map = _build_parent_map(tree)

    parents = parent_map.get(node_id, [])
    if not parents:
        # Root node
        return "pending"

    # Check each parent edge
    for parent_id, branch in parents:
        pstate = _get_node_state(tree, parent_id)
        if pstate.status == "done" and pstate.outcome == branch:
            return "pending"

    # Check if any parent is done but on wrong branch → skipped
    all_parents_decided = True
    for parent_id, branch in parents:
        pstate = _get_node_state(tree, parent_id)
        if pstate.status == "done":
            if pstate.outcome != branch:
                continue  # wrong branch
        else:
            all_parents_decided = False

    if all_parents_decided:
        return "skipped"

    return "locked"


# ---------------------------------------------------------------------------
# Success evaluation
# ---------------------------------------------------------------------------


def evaluate_node(tree: Tree, node_id: str) -> str | None:
    """Evaluate a node's success condition from campaign results.

    Returns ``"success"``, ``"failure"``, or ``None`` (no condition).
    """
    from retrain.status import scan_campaign

    node = tree.node_map[node_id]
    ns = _get_node_state(tree, node_id)

    if node.success_condition is None:
        return None

    if not ns.campaign_dir:
        raise ValueError(f"Node {node_id!r}: no campaign_dir in state")

    campaign = scan_campaign(Path(ns.campaign_dir))
    if campaign is None:
        raise ValueError(
            f"Node {node_id!r}: cannot scan campaign at {ns.campaign_dir}"
        )

    metric = node.success_condition.metric
    values: list[float] = []
    for run in campaign.runs:
        if not run.completed:
            continue
        val = getattr(run, metric, None)
        if val is not None:
            values.append(float(val))

    if not values:
        raise ValueError(
            f"Node {node_id!r}: no completed runs with metric {metric!r}"
        )

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


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "pending": "[ ]",
    "locked": "[#]",
    "running": "[~]",
    "done": {
        "success": "[✓]",
        "failure": "[✗]",
    },
    "skipped": "[-]",
}


def _icon(status: str, outcome: str = "") -> str:
    if status == "done":
        return _STATUS_ICONS["done"].get(outcome, "[?]")
    return _STATUS_ICONS.get(status, "[?]")


def format_tree(tree: Tree) -> str:
    """Render the tree as an ASCII diagram."""
    parent_map = _build_parent_map(tree)
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
            ns = _get_node_state(tree, node_id)
            status = effective_status(node_id, tree, parent_map)
            icon = _icon(status, ns.outcome)
            lines.append(f"{prefix}{icon} {node_id}  (see above)")
            return
        visited.add(node_id)

        node = tree.node_map[node_id]
        ns = _get_node_state(tree, node_id)
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
    parent_map = _build_parent_map(tree)
    lines: list[str] = []

    for node in tree.nodes:
        status = effective_status(node.id, tree, parent_map)
        if status == "pending":
            ns = _get_node_state(tree, node.id)
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
    parent_map = _build_parent_map(tree)
    ns = _get_node_state(tree, node_id)
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
    parent_map = _build_parent_map(tree)
    nodes_out = []
    for node in tree.nodes:
        ns = _get_node_state(tree, node.id)
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


def reset_node(tree: Tree, node_id: str) -> None:
    """Reset a node to re-runnable state. Preserves annotations."""
    if node_id not in tree.node_map:
        raise KeyError(f"Unknown node: {node_id!r}")

    ns = _get_node_state(tree, node_id)
    ns.status = "pending"
    ns.outcome = ""
    ns.campaign_dir = ""
    ns.result = {}
    ns.started_at = ""
    ns.completed_at = ""
    # Preserve annotations — they're a lab notebook
    save_state(tree)
