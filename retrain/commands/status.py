"""`retrain status` and `retrain top` commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Dead/partial/done campaigns disappear from the default view after 24h idle.
_STALE_HIDE_SECONDS = 86400


def run(args: list[str]) -> None:
    """Scan log directories and print run/campaign status."""
    import time as _time

    from retrain.status import (
        format_campaign,
        format_run,
        format_summary_banner,
        scan_all,
    )

    fmt = "text"
    root = "logs"
    show_all = False
    watch = False
    positional: list[str] = []
    for arg in args:
        if arg == "--json":
            fmt = "json"
        elif arg == "--all":
            show_all = True
        elif arg == "--active":
            show_all = False
        elif arg == "--watch":
            watch = True
        elif arg.startswith("--"):
            print(f"Unknown status flag: {arg}", file=sys.stderr)
            sys.exit(1)
        else:
            positional.append(arg)

    if positional:
        root = positional[0]

    root_path = Path(root)
    if not root_path.is_dir():
        print(f"No log directory found: {root}")
        sys.exit(1)

    while True:
        now = _time.time()
        runs, campaigns = scan_all(root_path)

        # Banner uses ALL campaigns (before filtering)
        banner = format_summary_banner(campaigns)

        if not show_all:

            from retrain.status import CampaignSummary

            def _is_visible(c: CampaignSummary) -> bool:
                if c.status == "running":
                    return True
                if c.status in ("dead", "partial", "done"):
                    # Show only if there was recent activity
                    return c.last_activity > now - _STALE_HIDE_SECONDS
                return True

            campaigns = [c for c in campaigns if _is_visible(c)]

        if fmt == "json":
            payload = {
                "root": str(root_path),
                "runs": [r.to_dict() for r in runs],
                "campaigns": [c.to_dict() for c in campaigns],
            }
            print(json.dumps(payload, indent=2))
            if not watch:
                return
        else:
            print(banner)
            print()
            if not runs and not campaigns:
                if show_all:
                    print(f"No runs or campaigns found in {root}")
                else:
                    print(f"No active campaigns in {root}  (use --all to see everything)")
            else:
                if campaigns:
                    for camp in campaigns:
                        print(format_campaign(camp))
                        print()

                if runs:
                    print("Standalone runs:")
                    for one_run in runs:
                        print(format_run(one_run))

        if not watch:
            return

        try:
            _time.sleep(5)
            # Clear screen for refresh
            print("\033[2J\033[H", end="")
        except KeyboardInterrupt:
            return


def run_top(args: list[str]) -> None:
    """Live dashboard: alias for ``retrain status --watch --active``."""
    status_args = ["--watch", "--active"]
    # Accept optional positional logdir
    for arg in args:
        if not arg.startswith("-"):
            status_args.append(arg)
            break
    run(status_args)
