"""Process-level telemetry."""

from __future__ import annotations

import sys


def max_rss_mb() -> float | None:
    """Return best-effort process peak RSS in MiB."""
    try:
        import resource
    except ImportError:
        return None

    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if raw_rss <= 0:
        return 0.0
    if sys.platform == "darwin":
        return raw_rss / (1024.0 * 1024.0)
    return raw_rss / 1024.0
