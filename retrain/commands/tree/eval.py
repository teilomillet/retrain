"""Tree node evaluation command helper."""

from __future__ import annotations


def evaluate_done(tree) -> None:
    """Evaluate success conditions for all done nodes without outcomes."""
    from retrain.tree.eval import evaluate_node

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
                    parts = [f"{key}={value}" for key, value in ns.result.items()]
                    result_str = f" ({', '.join(parts)})"
                print(f"  {node.id}: {outcome}{result_str}")
                evaluated += 1
            except Exception as exc:
                print(f"  {node.id}: error — {exc}")

    if evaluated == 0:
        print("No nodes to evaluate.")
