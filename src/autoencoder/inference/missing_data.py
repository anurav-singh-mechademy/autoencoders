"""Handle missing data in inference windows: forward-fill, null detection, partial-window flags."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_nulls(window: np.ndarray | pd.DataFrame) -> dict:
    """Check a window for null/NaN values.

    Returns:
        Dict with keys:
            null_count: total nulls
            null_pct_per_sensor: array of null % per sensor
            sensors_above_threshold: list of sensor indices exceeding max_null_pct
            has_nulls: bool
    """
    if isinstance(window, pd.DataFrame):
        values = window.values
    else:
        values = window

    null_mask = np.isnan(values)
    null_count = int(null_mask.sum())
    n_rows = values.shape[0]
    null_pct_per_sensor = (null_mask.sum(axis=0) / n_rows) * 100

    return {
        "null_count": null_count,
        "null_pct_per_sensor": null_pct_per_sensor,
        "has_nulls": null_count > 0,
    }


def forward_fill(window: np.ndarray, max_consecutive: int = 3) -> tuple[np.ndarray, dict]:
    """Forward-fill NaN values in a window, up to max_consecutive consecutive nulls.

    Args:
        window: Array of shape (n_rows, n_sensors).
        max_consecutive: Maximum consecutive NaN values to forward-fill.

    Returns:
        (filled_window, info) where info contains fill statistics.
    """
    filled = window.copy()
    n_rows, n_sensors = filled.shape
    total_filled = 0
    unfillable = 0

    for col in range(n_sensors):
        consecutive = 0
        last_valid = np.nan

        for row in range(n_rows):
            if np.isnan(filled[row, col]):
                consecutive += 1
                if consecutive <= max_consecutive and not np.isnan(last_valid):
                    filled[row, col] = last_valid
                    total_filled += 1
                else:
                    unfillable += 1
            else:
                last_valid = filled[row, col]
                consecutive = 0

    return filled, {
        "total_filled": total_filled,
        "unfillable": unfillable,
        "still_has_nulls": bool(np.isnan(filled).any()),
    }


def assess_window_quality(
    window: np.ndarray,
    max_null_pct_per_sensor: float = 5.0,
    max_consecutive_nulls: int = 3,
) -> dict:
    """Assess whether a window is usable for inference.

    Returns:
        Dict with:
            usable: bool — whether the window can be used after filling
            filled_window: the forward-filled window (or None if not usable)
            quality_flags: list of issues found
            fill_info: stats from forward-fill
    """
    null_info = check_nulls(window)

    if not null_info["has_nulls"]:
        return {
            "usable": True,
            "filled_window": window,
            "quality_flags": [],
            "fill_info": {"total_filled": 0, "unfillable": 0, "still_has_nulls": False},
        }

    filled, fill_info = forward_fill(window, max_consecutive=max_consecutive_nulls)
    flags = []

    # Check per-sensor null percentage (on original window)
    bad_sensors = np.where(null_info["null_pct_per_sensor"] > max_null_pct_per_sensor)[0]
    if len(bad_sensors) > 0:
        flags.append(f"sensors_exceed_null_threshold: {bad_sensors.tolist()}")

    if fill_info["still_has_nulls"]:
        flags.append("unfillable_nulls_remain")

    usable = not fill_info["still_has_nulls"] and len(bad_sensors) == 0

    return {
        "usable": usable,
        "filled_window": filled if usable else None,
        "quality_flags": flags,
        "fill_info": fill_info,
    }
