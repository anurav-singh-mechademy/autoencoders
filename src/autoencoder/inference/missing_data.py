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
    max_null_dominant_sensor_pct: float = 30.0,
) -> dict:
    """Assess whether a window is usable for inference, masking out individual
    null-dominant sensors instead of rejecting the whole window where possible.

    A sensor is "null-dominant" within this window if either:
      - its raw null fraction (before any fill) exceeds max_null_pct_per_sensor,
        even if every gap is individually short enough to forward-fill -- too
        much of that sensor's reading in this window would be fabricated to
        trust it, or
      - a gap longer than max_consecutive_nulls left it with unfillable NaNs.

    The window as a whole is rejected outright only if the *fraction of all
    sensors* that are null-dominant exceeds max_null_dominant_sensor_pct --
    below that, the window is still scored, with null-dominant sensors
    excluded from the reconstruction error and diagnosis (see
    src/autoencoder/inference/pipeline.py) rather than contaminating either
    with fabricated or missing values.

    Returns:
        Dict with:
            usable: bool — whether the window can be scored (possibly with
                some sensors masked out)
            filled_window: the forward-filled window (or None if not usable)
            quality_flags: list of issues found
            fill_info: stats from forward-fill
            masked_sensors: sorted list of null-dominant sensor indices to
                exclude from scoring (empty list if none)
    """
    n_sensors = window.shape[1]
    null_info = check_nulls(window)

    if not null_info["has_nulls"]:
        return {
            "usable": True,
            "filled_window": window,
            "quality_flags": [],
            "fill_info": {"total_filled": 0, "unfillable": 0, "still_has_nulls": False},
            "masked_sensors": [],
        }

    filled, fill_info = forward_fill(window, max_consecutive=max_consecutive_nulls)

    bad_pct_sensors = set(np.where(null_info["null_pct_per_sensor"] > max_null_pct_per_sensor)[0].tolist())
    still_null_sensors = set(np.where(np.isnan(filled).any(axis=0))[0].tolist())
    null_dominant_sensors = sorted(bad_pct_sensors | still_null_sensors)

    null_dominant_pct = len(null_dominant_sensors) / n_sensors * 100
    flags = []
    if null_dominant_sensors:
        flags.append(f"null_dominant_sensors ({null_dominant_pct:.1f}% of sensors): {null_dominant_sensors}")

    if null_dominant_pct > max_null_dominant_sensor_pct:
        flags.append(
            f"null_dominant_sensor_pct {null_dominant_pct:.1f}% exceeds window "
            f"threshold {max_null_dominant_sensor_pct:.1f}% -- window rejected"
        )
        return {
            "usable": False,
            "filled_window": None,
            "quality_flags": flags,
            "fill_info": fill_info,
            "masked_sensors": null_dominant_sensors,
        }

    return {
        "usable": True,
        "filled_window": filled,
        "quality_flags": flags,
        "fill_info": fill_info,
        "masked_sensors": null_dominant_sensors,
    }
