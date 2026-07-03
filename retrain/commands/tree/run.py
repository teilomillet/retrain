"""`retrain tree` command."""

from __future__ import annotations

import json
import sys

from retrain.commands.tree.eval import evaluate_done
from retrain.commands.tree.node import run_campaign_node
from retrain.commands.tree.note import add_note

USAGE = (
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
        print(USAGE)
        return

    json_flag = "--json" in args
    if json_flag:
        args = [arg for arg in args if arg != "--json"]

    subcommands = {"next", "run", "note", "eval", "show", "reset"}
    if args and args[0] in subcommands:
        subcmd = args[0]
        rest = args[1:]
    else:
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
            print(USAGE)
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
            print(USAGE)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        run_campaign_node(tree, node_id)
        return

    if subcmd == "note":
        if len(rest) < 2:
            print("Error: 'note' requires a node id and text.")
            print(USAGE)
            sys.exit(1)
        node_id = rest[0]
        text = rest[1]
        tree_path = rest[2] if len(rest) > 2 else "tree.toml"
        tree = load_tree(tree_path)
        add_note(tree, node_id, text)
        return

    if subcmd == "eval":
        tree_path = rest[0] if rest else "tree.toml"
        tree = load_tree(tree_path)
        evaluate_done(tree)
        return

    if subcmd == "reset":
        if not rest:
            print("Error: 'reset' requires a node id.")
            print(USAGE)
            sys.exit(1)
        node_id = rest[0]
        tree_path = rest[1] if len(rest) > 1 else "tree.toml"
        tree = load_tree(tree_path)
        reset_node(tree, node_id)
        print(f"Node {node_id!r} reset to pending.")
        return

    print(f"Unknown tree subcommand: {subcmd}")
    print(USAGE)
    sys.exit(1)
