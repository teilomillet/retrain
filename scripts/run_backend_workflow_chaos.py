from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ordeal.explore import Explorer

from tests.test_ordeal_backend_workflow import BackendWorkflowChaos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-budget Ordeal gate for the backend workflow chaos model."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--steps-per-run", type=int, default=80)
    parser.add_argument("--min-edges", type=int, default=800)
    parser.add_argument("--min-states", type=int, default=700)
    parser.add_argument("--min-line-coverage", type=float, default=0.72)
    parser.add_argument("--min-properties", type=int, default=6)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    explorer = Explorer(
        BackendWorkflowChaos,
        target_modules=["tests.test_ordeal_backend_workflow"],
        seed=args.seed,
        rule_swarm=True,
        workers=1,
        record_traces=False,
        ngram=2,
    )
    result = explorer.run(
        max_runs=args.max_runs,
        max_time=args.max_time,
        steps_per_run=args.steps_per_run,
        shrink=False,
    )

    print(result.summary())

    issues: list[str] = []
    if result.failures:
        issues.append(f"found {len(result.failures)} failure(s)")
    if result.unique_edges < args.min_edges:
        issues.append(
            f"edge coverage too low: {result.unique_edges} < {args.min_edges}"
        )
    if result.unique_states < args.min_states:
        issues.append(
            f"state coverage too low: {result.unique_states} < {args.min_states}"
        )

    line_coverage = 0.0
    if result.lines_total > 0:
        line_coverage = result.lines_covered / result.lines_total
    if line_coverage < args.min_line_coverage:
        issues.append(
            "line coverage too low: "
            f"{line_coverage:.1%} < {args.min_line_coverage:.1%}"
        )

    if result.properties_satisfied < args.min_properties:
        issues.append(
            "property coverage too low: "
            f"{result.properties_satisfied} < {args.min_properties}"
        )

    if issues:
        print("\nBackend workflow chaos gate failed:")
        for issue in issues:
            print(f"- {issue}")
        for failure in result.failures[:5]:
            print(f"- failure: {failure}")
        return 1

    print(
        "\nBackend workflow chaos gate passed: "
        f"{result.unique_edges} edges, "
        f"{result.unique_states} states, "
        f"{line_coverage:.1%} line coverage, "
        f"{result.properties_satisfied} properties."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
