#!/usr/bin/env python3
"""Wrapper script for Delight campaign summaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrain.delight_campaign_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
