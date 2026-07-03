"""Experiment tree data model."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from pathlib import Path

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


def node_state(tree: Tree, node_id: str) -> NodeState:
    """Get or create runtime state for a node."""
    if node_id not in tree.state.nodes:
        tree.state.nodes[node_id] = NodeState()
    return tree.state.nodes[node_id]
