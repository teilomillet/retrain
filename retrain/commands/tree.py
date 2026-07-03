"""`retrain tree` command."""

from __future__ import annotations

import json
import sys

_USAGE = (
    "Usage:\n"
    "  retrain tree [tree.toml] [--json]          — view tree\n"
    "  retrain tree next [tree.toml]              — show pending nodes\n"
    "  retrain tree show <node> [tree.toml] [--json] — node details\n"
    "  retrain tree run <node> [tree.toml]        — launch node's campaign\n"
    '  retrain tree note <node> "text" [tree.toml] — add annotation\n'
    "  retrain tree eval [tree.toml]              — evaluate success conditions\n"
    "  retrain tree reset <node> [tree.toml]      — reset node to pending\n"
)


def run(args: list[str]) -> None:
    """Experiment tech-tree commands."""
    from retrain.tree import (
        format_next,
        format_show,
        format_tree,
        format_tree_json,
        load_tree,
        reset_node,
    )

    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return

    # Extract --json flag
    json_flag = "--json" in args
    if json_flag:
        args = [a for a in args if a != "--json"]

    # Determine subcommand vs bare tree path
    subcommands = {"next", "run", "note", "eval", "show", "reset"}
    if args and args[0] in subcommands:
        subcmd = args[0]
        rest = args[1:]
    else:
        # Default: view
        subcmd = "view"
        rest = args

    if subcmd == "view":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        if json_flag:
            print(json.dumps(format_tree_json(tree), indent=2))
        else:
            print(format_tree(tree))
        return

    if subcmd == "next":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        print(format_next(tree))
        return

    if subcmd == "show":
        if not rest:
            print("Error: 'show' requires a node id.")
            print(_USAGE)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        try:
            if json_flag:
                data = format_tree_json(tree)
                node_data = [n for n in data["nodes"] if n["id"] == node_id]
                if not node_data:
                    raise KeyError(node_id)
                print(json.dumps(node_data[0], indent=2))
            else:
                print(format_show(tree, node_id))
        except KeyError:
            print(f"Error: unknown node {node_id!r}")
            sys.exit(1)
        return

    if subcmd == "run":
        if not rest:
            print("Error: 'run' requires a node id.")
            print(_USAGE)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        _run_node(tree, node_id)
        return

    if subcmd == "note":
        if len(rest) < 2:
            print("Error: 'note' requires a node id and text.")
            print(_USAGE)
            sys.exit(1)
        node_id = rest[0]
        text = rest[1]
        tree_path = rest[2] if len(rest) > 2 else "tree.toml"
        tree = load_tree(tree_path)
        _add_note(tree, node_id, text)
        return

    if subcmd == "eval":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        _eval_nodes(tree)
        return

    if subcmd == "reset":
        if not rest:
            print("Error: 'reset' requires a node id.")
            print(_USAGE)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        reset_node(tree, node_id)
        print(f"Node {node_id!r} reset to pending.")
        return

    print(f"Unknown tree subcommand: {subcmd}")
    print(_USAGE)
    sys.exit(1)


def _run_node(tree, node_id: str) -> None:
    """Launch a node's campaign and record state."""
    from datetime import datetime, timezone

    from retrain.campaign import run_campaign
    from retrain.tree import NodeState, save_state

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


def _add_note(tree, node_id: str, text: str) -> None:
    """Add an annotation to a node."""
    from datetime import datetime, timezone

    from retrain.tree import Annotation, NodeState, save_state

    if node_id not in tree.node_map:
        print(f"Error: unknown node {node_id!r}")
        sys.exit(1)

    ns = tree.state.nodes.setdefault(node_id, NodeState())
    ns.annotations.append(
        Annotation(text=text, at=datetime.now(timezone.utc).isoformat())
    )
    save_state(tree)
    print(f"Note added to {node_id!r}.")


def _eval_nodes(tree) -> None:
    """Evaluate success conditions for all done nodes without outcomes."""
    from retrain.tree import evaluate_node

    evaluated = 0
    for node in tree.nodes:
        ns = tree.state.nodes.get(node.id)
        if ns and ns.status == "done" and not ns.outcome:
            if node.success_condition is None:
                print(f"  {node.id}: no success condition defined")
                continue
            try:
                outcome = evaluate_node(tree, node.id)
                result_str = ""
                if ns.result:
                    parts = [f"{k}={v}" for k, v in ns.result.items()]
                    result_str = f" ({', '.join(parts)})"
                print(f"  {node.id}: {outcome}{result_str}")
                evaluated += 1
            except Exception as e:
                print(f"  {node.id}: error — {e}")

    if evaluated == 0:
        print("No nodes to evaluate.")
