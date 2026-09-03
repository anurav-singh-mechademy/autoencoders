"""Weekly health metrics logging per equipment unit."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_weekly_metrics(
    daily_scores: list[float],
    daily_zones: list[str],
) -> dict:
    """Compute summary metrics for a week of monitoring data.

    Args:
        daily_scores: List of daily average anomaly scores (up to 7).
        daily_zones: List of daily zone classifications.

    Returns:
        Dict with mean/max/min scores, zone counts, alert rate.
    """
    scores = np.array(daily_scores, dtype=float)

    zone_counts = {"green": 0, "yellow": 0, "red": 0}
    for z in daily_zones:
        if z in zone_counts:
            zone_counts[z] += 1

    total = len(daily_zones) if daily_zones else 1
    alert_rate = (zone_counts["yellow"] + zone_counts["red"]) / total

    return {
        "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
        "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0,
        "min_score": float(np.min(scores)) if len(scores) > 0 else 0.0,
        "std_score": float(np.std(scores)) if len(scores) > 0 else 0.0,
        "zone_counts": zone_counts,
        "alert_rate": float(alert_rate),
        "n_days": len(daily_scores),
    }


def log_weekly_health(
    equipment_id: str,
    week_start: str,
    metrics: dict,
    log_dir: str | Path,
) -> Path:
    """Append weekly health metrics to a JSONL log file.

    Args:
        equipment_id: Equipment identifier.
        week_start: ISO date string for the start of the week.
        metrics: Output from compute_weekly_metrics().
        log_dir: Directory to store log files.

    Returns:
        Path to the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{equipment_id}_health.jsonl"

    entry = {
        "equipment_id": equipment_id,
        "week_start": week_start,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Logged weekly health for %s (week %s)", equipment_id, week_start)
    return log_file


def read_health_log(log_path: str | Path) -> list[dict]:
    """Read all entries from a health log JSONL file."""
    entries = []
    path = Path(log_path)
    if not path.exists():
        return entries

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    return entries
