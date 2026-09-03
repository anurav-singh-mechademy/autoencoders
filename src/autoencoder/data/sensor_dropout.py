"""Sensor dropout filtering -- excludes windows with too many null/stuck sensors."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorDropoutResult:
    """Result of sensor dropout detection."""

    is_dropped: np.ndarray        # Boolean mask -- True = window excluded
    n_dropped: int
    n_total: int
    bad_sensor_pct: np.ndarray    # Per-window fraction (%) of null/stuck sensors
    threshold_used: float


def detect_sensor_dropout_windows(
    windows: list[np.ndarray],
    threshold_pct: float = 5.0,
    stuck_atol: float = 1e-9,
) -> SensorDropoutResult:
    """Flag windows where too many sensors are null or stuck.

    A sensor is considered null/stuck within a window if either:
      - any value for that sensor in the window is NaN, or
      - the sensor never changes across the window (max == min within
        stuck_atol), i.e. it reads as flat/frozen.

    A window is excluded if the fraction of null/stuck sensors exceeds
    threshold_pct (default 5%, per spec).

    Args:
        windows: List of (window_size, n_sensors) arrays.
        threshold_pct: Percentage of sensors above which a window is excluded.
        stuck_atol: Absolute tolerance for detecting a "stuck" (flat) sensor.

    Returns:
        SensorDropoutResult with boolean mask and diagnostics.
    """
    n_total = len(windows)
    bad_sensor_pct = np.zeros(n_total)

    for i, w in enumerate(windows):
        n_sensors = w.shape[1]
        is_null = np.isnan(w).any(axis=0)
        # A sensor that is entirely NaN has no defined range; is_null already
        # covers it, so suppress the resulting "All-NaN slice" warning rather
        # than let it mask a real signal elsewhere.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            is_stuck = (np.nanmax(w, axis=0) - np.nanmin(w, axis=0)) <= stuck_atol
        is_stuck = np.nan_to_num(is_stuck, nan=0.0).astype(bool)
        n_bad = int(np.sum(is_null | is_stuck))
        bad_sensor_pct[i] = (n_bad / n_sensors) * 100.0

    is_dropped = bad_sensor_pct > threshold_pct
    n_dropped = int(np.sum(is_dropped))

    logger.info(
        "Sensor dropout filter: %d / %d windows excluded (>%.1f%% null/stuck sensors).",
        n_dropped, n_total, threshold_pct,
    )

    return SensorDropoutResult(
        is_dropped=is_dropped,
        n_dropped=n_dropped,
        n_total=n_total,
        bad_sensor_pct=bad_sensor_pct,
        threshold_used=threshold_pct,
    )


def remove_sensor_dropout_windows(
    windows: list[np.ndarray],
    result: SensorDropoutResult,
) -> list[np.ndarray]:
    """Remove windows flagged for excessive null/stuck sensors."""
    clean = [w for w, dropped in zip(windows, result.is_dropped) if not dropped]
    logger.info(
        "Removed %d sensor-dropout windows. %d remaining.",
        result.n_dropped, len(clean),
    )
    return clean
