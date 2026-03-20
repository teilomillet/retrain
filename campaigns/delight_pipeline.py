#!/usr/bin/env python3
"""Thin pipeline entrypoint for the Delight campaign sweep."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrain.campaign import run_campaign
from retrain.delight_campaign_summary import summarize_delight_campaign, write_delight_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or summarize the Delight campaign sweep."
    )
    parser.add_argument(
        "--campaign",
        default=str(Path(__file__).with_name("delight-gate.toml")),
        help="Campaign TOML to run",
    )
    parser.add_argument(
        "--summary-only",
        default="",
        help="Existing campaign directory to summarize without rerunning training",
    )
    args = parser.parse_args(argv)

    os.chdir(ROOT)

    if args.summary_only:
        campaign_dir = Path(args.summary_only)
        summary = summarize_delight_campaign(campaign_dir)
        json_path, md_path = write_delight_summary(campaign_dir, summary)
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        return 0

    campaign_dir = run_campaign(args.campaign)
    print(f"Delight campaign complete: {campaign_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
